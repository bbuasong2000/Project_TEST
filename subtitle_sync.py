#!/usr/bin/env python3
"""폴더 안의 영상/자막 쌍을 찾아 싱크를 점검하고 맞추는 스크립트.

영상의 음성에서 대사가 있는 구간을 찾고, 자막이 표시되는 구간과 비교해서
자막을 얼마나 옮겨야 하는지(시간 이동)와 프레임레이트 차이(비율)를 계산한다.

지원 자막: .smi / .srt / .ass / .ssa
필요 프로그램: Python 3.8+, numpy, ffmpeg (PATH에 등록)

사용 예:
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난" --check
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난"
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난" --split
"""

import argparse
import bisect
import csv
import re
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy가 필요합니다. 'pip install numpy' 로 설치해 주세요.")


VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".wmv", ".mov", ".m4v", ".ts", ".mpg", ".mpeg", ".webm", ".flv"}
SUB_EXTS = {".smi", ".srt", ".ass", ".ssa"}

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

# 흔한 프레임레이트 차이 (23.976 / 24 / 25 / 29.97 fps 사이 변환)
FPS_RATIOS = [
    1.0,
    25 / 23.976, 23.976 / 25,
    24 / 23.976, 23.976 / 24,
    25 / 24, 24 / 25,
    29.97 / 25, 25 / 29.97,
]

# 피크 신뢰도(PSR) 기준
PSR_GOOD = 8.0
PSR_MIN = 5.0
# 이보다 작은 차이는 "정상"으로 본다
OK_OFFSET_MS = 100
# 구간별 편차가 이보다 크면 경고한다
SPREAD_WARN_MS = 500


# ---------------------------------------------------------------------------
# 자막 읽기 / 쓰기
# ---------------------------------------------------------------------------

def read_text(path):
    """BOM, UTF-8, CP949 순서로 인코딩을 판별해서 읽는다. (텍스트, 인코딩) 반환."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("cp949", errors="replace"), "cp949"


def write_text(path, text, encoding):
    """원본과 같은 인코딩으로 저장한다. 표현할 수 없는 문자가 있으면 UTF-8(BOM)로 저장."""
    try:
        data = text.encode(encoding)
    except UnicodeEncodeError:
        encoding = "utf-8-sig"
        data = text.encode(encoding)
    path.write_bytes(data)
    return encoding


SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})(\s*-->\s*)(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
SMI_SYNC_RE = re.compile(r"(<sync\b[^>]*?\bstart\s*=\s*[\"']?)(-?\d+)", re.IGNORECASE)
SMI_END_RE = re.compile(r"(<sync\b[^>]*?\bend\s*=\s*[\"']?)(-?\d+)", re.IGNORECASE)
SMI_BLOCK_RE = re.compile(r"<sync\b[^>]*?\bstart\s*=\s*[\"']?(-?\d+)[^>]*>(.*?)(?=<sync\b|</body|$)",
                          re.IGNORECASE | re.DOTALL)
ASS_DIALOGUE_RE = re.compile(
    r"^(Dialogue:\s*[^,]*,)(\d+):(\d{2}):(\d{2})\.(\d{2}),(\d+):(\d{2}):(\d{2})\.(\d{2}),",
    re.MULTILINE,
)
TAG_RE = re.compile(r"<[^>]*>")


def ms_from_parts(h, m, s, frac, frac_digits):
    frac_ms = int(frac) * (10 ** (3 - frac_digits))
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + frac_ms


def srt_stamp(ms):
    ms = max(0, int(round(ms)))
    h, rest = divmod(ms, 3600000)
    m, rest = divmod(rest, 60000)
    s, rest = divmod(rest, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, rest)


def ass_stamp(ms):
    cs = max(0, int(round(ms / 10.0)))
    h, rest = divmod(cs, 360000)
    m, rest = divmod(rest, 6000)
    s, rest = divmod(rest, 100)
    return "%d:%02d:%02d.%02d" % (h, m, s, rest)


def parse_cues(text, ext):
    """자막의 표시 구간 목록 [(시작ms, 끝ms), ...] 을 반환한다."""
    cues = []
    if ext == ".srt":
        for m in SRT_TIME_RE.finditer(text):
            start = ms_from_parts(m.group(1), m.group(2), m.group(3), m.group(4), len(m.group(4)))
            end = ms_from_parts(m.group(6), m.group(7), m.group(8), m.group(9), len(m.group(9)))
            if end > start:
                cues.append((start, end))
    elif ext in (".ass", ".ssa"):
        for m in ASS_DIALOGUE_RE.finditer(text):
            start = ms_from_parts(m.group(2), m.group(3), m.group(4), m.group(5), 2)
            end = ms_from_parts(m.group(6), m.group(7), m.group(8), m.group(9), 2)
            if end > start:
                cues.append((start, end))
    elif ext == ".smi":
        blocks = []
        for m in SMI_BLOCK_RE.finditer(text):
            body = TAG_RE.sub("", m.group(2)).replace("&nbsp;", " ").strip()
            blocks.append((int(m.group(1)), body))
        blocks.sort(key=sync_start)
        for i in range(len(blocks) - 1):
            start, body = blocks[i]
            end = blocks[i + 1][0]
            if body and end > start:
                # 다음 SYNC까지 너무 길면(대사 없는 긴 공백) 최대 10초로 자른다
                cues.append((start, min(end, start + 10000)))
        if blocks and blocks[-1][1]:
            cues.append((blocks[-1][0], blocks[-1][0] + 3000))
    # 여러 언어/스타일이 겹치는 경우를 위해 정렬
    cues.sort()
    return cues


def sync_start(block):
    return block[0]


class TimeMap:
    """원래 자막 시간(ms) -> 보정된 시간(ms). 구간별로 다른 오프셋을 가질 수 있다."""

    def __init__(self, ratio, offsets):
        # offsets: [(이 시간 이후부터 적용, 오프셋ms), ...] 시간 순 정렬
        self.ratio = ratio
        self.bounds = [b for b, _ in offsets]
        self.offsets = [o for _, o in offsets]

    def __call__(self, ms):
        idx = bisect.bisect_right(self.bounds, ms) - 1
        if idx < 0:
            idx = 0
        return max(0.0, ms * self.ratio + self.offsets[idx])


def apply_map(text, ext, tmap):
    """원본 텍스트의 시간 값만 바꿔서 반환한다. 나머지 내용은 그대로 유지."""
    if ext == ".srt":
        def fix_srt(m):
            start = ms_from_parts(m.group(1), m.group(2), m.group(3), m.group(4), len(m.group(4)))
            end = ms_from_parts(m.group(6), m.group(7), m.group(8), m.group(9), len(m.group(9)))
            # 끝 시간은 시작 시간 기준의 오프셋을 따라가야 구간 경계에서 뒤집히지 않는다
            new_start = tmap(start)
            new_end = new_start + (end - start) * tmap.ratio
            return srt_stamp(new_start) + m.group(5) + srt_stamp(new_end)
        return SRT_TIME_RE.sub(fix_srt, text)

    if ext in (".ass", ".ssa"):
        def fix_ass(m):
            start = ms_from_parts(m.group(2), m.group(3), m.group(4), m.group(5), 2)
            end = ms_from_parts(m.group(6), m.group(7), m.group(8), m.group(9), 2)
            new_start = tmap(start)
            new_end = new_start + (end - start) * tmap.ratio
            return m.group(1) + ass_stamp(new_start) + "," + ass_stamp(new_end) + ","
        return ASS_DIALOGUE_RE.sub(fix_ass, text)

    if ext == ".smi":
        def fix_smi(m):
            return m.group(1) + str(int(round(tmap(int(m.group(2))))))
        text = SMI_SYNC_RE.sub(fix_smi, text)
        return SMI_END_RE.sub(fix_smi, text)

    raise ValueError("지원하지 않는 자막 형식: " + ext)


# ---------------------------------------------------------------------------
# 음성 분석
# ---------------------------------------------------------------------------

def extract_audio(video):
    """ffmpeg로 첫 번째 오디오 트랙을 16kHz 모노로 읽는다."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        sys.exit("ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하고 PATH에 등록해 주세요.")
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError("오디오 추출 실패: " + (msg[-300:] or "오디오 트랙 없음"))
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)


def speech_signal(samples):
    """10ms 프레임마다 음성 대역(300~3400Hz) 에너지를 구해 정규화한 신호."""
    n_frames = len(samples) // FRAME_LEN
    frames = samples[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
    frames = frames * np.hanning(FRAME_LEN).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME_LEN, 1.0 / SAMPLE_RATE)
    band = (freqs >= 300) & (freqs <= 3400)
    energy = np.log10(spec[:, band].sum(axis=1) + 1e-6)
    # 짧은 순간 잡음을 줄이기 위해 50ms 이동 평균
    kernel = np.ones(5, dtype=np.float32) / 5
    energy = np.convolve(energy, kernel, mode="same")
    # 무음 구간 기준으로 정규화하고, 큰 값은 잘라서 효과음 영향을 줄인다
    floor = np.percentile(energy, 20)
    top = np.percentile(energy, 95)
    sig = np.clip((energy - floor) / max(top - floor, 1e-6), 0.0, 1.0)
    return sig - sig.mean()


def cue_signal(cues, length, ratio, window=None):
    """자막 표시 구간을 10ms 프레임 신호로 만든다. window=(시작ms, 끝ms)면 그 구간만."""
    sig = np.zeros(length, dtype=np.float32)
    for start, end in cues:
        if window is not None and not (window[0] <= start < window[1]):
            continue
        a = int(start * ratio / FRAME_MS)
        b = int(end * ratio / FRAME_MS)
        a = max(a, 0)
        b = min(b, length)
        if b > a:
            sig[a:b] = 1.0
    if window is None:
        return sig - sig.mean()
    a = max(int(window[0] * ratio / FRAME_MS), 0)
    b = min(int(window[1] * ratio / FRAME_MS), length)
    if b > a:
        sig[a:b] -= sig[a:b].mean()
    return sig


def correlate(audio_sig, sub_sig, max_lag, center=0):
    """lag(프레임) 별 상관값. 양수 lag = 자막을 뒤로 미뤄야 함. (lags, 값) 반환."""
    n = len(audio_sig) + len(sub_sig)
    size = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(audio_sig, size)
    fs = np.fft.rfft(sub_sig, size)
    corr = np.fft.irfft(fa * np.conj(fs), size)
    lags = np.arange(center - max_lag, center + max_lag + 1)
    values = corr[lags % size]
    norm = np.linalg.norm(audio_sig) * np.linalg.norm(sub_sig)
    if norm > 0:
        values = values / norm
    return lags, values


def peak_info(lags, values):
    """최고점 lag, 상관값, PSR(최고점이 주변보다 얼마나 두드러지는지)."""
    best = int(np.argmax(values))
    exclude = 100  # 최고점 주변 ±1초는 제외하고 배경 수준을 계산
    mask = np.ones(len(values), dtype=bool)
    mask[max(0, best - exclude): best + exclude + 1] = False
    side = values[mask]
    if len(side) < 10 or side.std() == 0:
        psr = 0.0
    else:
        psr = float((values[best] - side.mean()) / side.std())
    return int(lags[best]), float(values[best]), psr


def analyze(audio_sig, cues, max_offset_ms, try_ratios):
    """전체 자막에 대한 최적 (비율, 오프셋ms, 상관값, PSR)."""
    length = max(len(audio_sig), int(cues[-1][1] * 1.3 / FRAME_MS) + 1)
    audio = np.zeros(length, dtype=np.float32)
    audio[: len(audio_sig)] = audio_sig
    max_lag = max_offset_ms // FRAME_MS

    ratios = FPS_RATIOS if try_ratios else [1.0]
    best = None
    for ratio in ratios:
        sub = cue_signal(cues, length, ratio)
        lags, values = correlate(audio, sub, max_lag)
        lag, score, psr = peak_info(lags, values)
        if best is None or score > best[2]:
            best = (ratio, lag * FRAME_MS, score, psr)
    # 비율 1.0이 거의 같은 점수면 1.0을 우선 (잘못된 비율 선택 방지)
    if best[0] != 1.0:
        sub = cue_signal(cues, length, 1.0)
        lags, values = correlate(audio, sub, max_lag)
        lag, score, psr = peak_info(lags, values)
        if score >= best[2] * 0.97:
            best = (1.0, lag * FRAME_MS, score, psr)
    return best, audio


def segment_offsets(audio, cues, ratio, global_offset_ms, segment_ms, search_ms):
    """자막을 시간 구간으로 나눠 구간별 오프셋을 구한다.

    반환: [(구간 시작ms(원본 자막 기준), 오프셋ms, PSR, 자막 개수), ...]
    """
    length = len(audio)
    first = cues[0][0]
    last = cues[-1][0]
    results = []
    start = first
    center = global_offset_ms // FRAME_MS
    max_lag = search_ms // FRAME_MS
    while start <= last:
        window = (start, start + segment_ms)
        count = 0
        for cue in cues:
            if window[0] <= cue[0] < window[1]:
                count += 1
        if count >= 5:
            sub = cue_signal(cues, length, ratio, window)
            lags, values = correlate(audio, sub, max_lag, center)
            lag, _, psr = peak_info(lags, values)
            results.append((start, lag * FRAME_MS, psr, count))
        start += segment_ms
    return results


def piecewise_offsets(segments, global_offset_ms):
    """신뢰도 낮은 구간은 앞 구간 값을 이어받고, 짧게 튀는 값은 정리한다."""
    offsets = []
    prev = global_offset_ms
    for seg_start, off, psr, _ in segments:
        if psr < PSR_MIN:
            off = prev
        offsets.append([seg_start, off])
        prev = off
    # 양옆이 같은 값인데 혼자 다른 구간은 잘못 잡힌 것으로 보고 양옆 값으로 맞춘다
    for i in range(1, len(offsets) - 1):
        left = offsets[i - 1][1]
        right = offsets[i + 1][1]
        if abs(left - right) <= OK_OFFSET_MS and abs(offsets[i][1] - left) > SPREAD_WARN_MS:
            offsets[i][1] = left
    if not offsets:
        return [(0, global_offset_ms)]
    offsets[0][0] = 0
    # 오프셋이 같은 이웃 구간은 하나로 합친다
    result = []
    for seg_start, off in offsets:
        if result and result[-1][1] == off:
            continue
        result.append((seg_start, off))
    return result


def cue_score(audio, cue, ratio, offset_ms):
    """오프셋을 적용했을 때 그 대사 구간의 평균 음성 신호 값."""
    a = int((cue[0] * ratio + offset_ms) / FRAME_MS)
    b = int((cue[1] * ratio + offset_ms) / FRAME_MS)
    a = max(a, 0)
    b = min(b, len(audio))
    if b <= a:
        return 0.0
    return float(audio[a:b].mean())


def refine_bounds(audio, cues, ratio, pieces):
    """구간 경계를 대사 단위로 다시 정한다. (경계가 구간 한가운데 있을 때 대비)"""
    if len(pieces) < 2:
        return pieces
    starts = [c[0] for c in cues]
    refined = [list(p) for p in pieces]
    for i in range(1, len(refined)):
        before_off = refined[i - 1][1]
        after_off = refined[i][1]
        lo = refined[i - 1][0]
        hi = refined[i + 1][0] if i + 1 < len(refined) else float("inf")
        first = bisect.bisect_left(starts, lo)
        last = bisect.bisect_left(starts, hi)
        if last - first < 2:
            continue
        # 앞쪽은 before_off, 뒤쪽은 after_off 가 가장 잘 맞는 분할 지점을 찾는다
        diffs = []
        for k in range(first, last):
            diffs.append(cue_score(audio, cues[k], ratio, after_off)
                         - cue_score(audio, cues[k], ratio, before_off))
        total_after = sum(diffs)
        best_split = first
        best_gain = total_after
        running = 0.0
        for j in range(len(diffs)):
            running += diffs[j]
            gain = total_after - running
            if gain > best_gain:
                best_gain = gain
                best_split = first + j + 1
        if best_split >= len(cues):
            best_split = len(cues) - 1
        refined[i][0] = max(starts[best_split], refined[i - 1][0] + 1)
    result = []
    for start, off in refined:
        result.append((start, off))
    return result


# ---------------------------------------------------------------------------
# 파일 짝 찾기 / 실행
# ---------------------------------------------------------------------------

def find_pairs(folder, recursive, suffix):
    pattern = "**/*" if recursive else "*"
    videos = {}
    subs = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in VIDEO_EXTS:
            videos[(path.parent, path.stem.lower())] = path
        elif ext in SUB_EXTS and not path.stem.lower().endswith(suffix.lower()):
            subs.append(path)

    pairs = []
    unmatched = []
    for sub in subs:
        stem = sub.stem.lower()
        video = videos.get((sub.parent, stem))
        # "영상이름.ko.smi" / "영상이름.kor.srt" 같은 형태도 짝으로 인정
        while video is None and "." in stem:
            stem = stem.rsplit(".", 1)[0]
            video = videos.get((sub.parent, stem))
        if video is None:
            unmatched.append(sub)
        else:
            pairs.append((video, sub))
    return pairs, unmatched


def process(video, sub, args):
    row = {
        "영상": video.name, "자막": sub.name, "판정": "", "오프셋(초)": "", "비율": "",
        "신뢰도(PSR)": "", "구간별 오프셋(초)": "", "결과 파일": "", "메모": "",
    }
    text, encoding = read_text(sub)
    ext = sub.suffix.lower()
    cues = parse_cues(text, ext)
    if len(cues) < 10:
        row["판정"] = "건너뜀"
        row["메모"] = "자막 대사를 읽지 못함 (대사 %d개)" % len(cues)
        return row

    audio_sig = speech_signal(extract_audio(video))
    (ratio, offset_ms, score, psr), audio = analyze(
        audio_sig, cues, args.max_offset * 1000, not args.no_ratio)
    segments = segment_offsets(audio, cues, ratio, offset_ms,
                               args.segment * 1000, args.segment_search * 1000)

    row["오프셋(초)"] = "%+.2f" % (offset_ms / 1000.0)
    row["비율"] = "%.5f" % ratio
    row["신뢰도(PSR)"] = "%.1f" % psr
    seg_text = []
    reliable = []
    for seg_start, off, seg_psr, _ in segments:
        mark = "" if seg_psr >= PSR_MIN else "?"
        seg_text.append("%d분:%+.2f%s" % (seg_start // 60000, off / 1000.0, mark))
        if seg_psr >= PSR_MIN:
            reliable.append(off)
    row["구간별 오프셋(초)"] = " ".join(seg_text)
    spread = (max(reliable) - min(reliable)) if len(reliable) >= 2 else 0

    if psr < PSR_MIN and not args.force:
        row["판정"] = "판단불가"
        row["메모"] = "음성/자막 일치도가 낮음. 수동 확인 필요 (--force로 강제 적용 가능)"
        return row

    use_split = args.split and spread > SPREAD_WARN_MS
    if use_split:
        pieces = refine_bounds(audio, cues, ratio, piecewise_offsets(segments, offset_ms))
        tmap = TimeMap(ratio, pieces)
    else:
        tmap = TimeMap(ratio, [(0, offset_ms)])

    needs_fix = use_split or ratio != 1.0 or abs(offset_ms) >= OK_OFFSET_MS
    if not needs_fix:
        row["판정"] = "정상"
    elif use_split:
        row["판정"] = "구간별 보정"
    else:
        row["판정"] = "보정 필요"

    notes = []
    if psr < PSR_GOOD:
        notes.append("신뢰도 보통, 결과 확인 권장")
    if spread > SPREAD_WARN_MS and not use_split:
        notes.append("구간별 차이 %.1f초 (중간 광고 컷 등). --split 사용 검토" % (spread / 1000.0))
    row["메모"] = "; ".join(notes)

    if needs_fix and not args.check:
        out = sub.with_name(sub.stem + args.suffix + sub.suffix)
        if out.exists() and not args.overwrite:
            row["메모"] = "; ".join(notes + ["결과 파일이 이미 있어 저장 안 함 (--overwrite)"])
            return row
        used = write_text(out, apply_map(text, ext, tmap), encoding)
        row["결과 파일"] = out.name
        if used != encoding:
            notes.append("인코딩 %s -> %s" % (encoding, used))
            row["메모"] = "; ".join(notes)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description="폴더 안의 영상/자막 싱크를 점검하고 맞춥니다.")
    parser.add_argument("folder", help="영상과 자막이 있는 폴더")
    parser.add_argument("--check", action="store_true", help="점검만 하고 파일은 만들지 않음")
    parser.add_argument("--split", action="store_true",
                        help="구간마다 어긋난 정도가 다르면(광고 컷 등) 구간별로 따로 보정")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더까지 검사")
    parser.add_argument("--suffix", default=".synced", help="결과 파일 이름에 붙일 말 (기본 .synced)")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 결과 파일을 덮어씀")
    parser.add_argument("--force", action="store_true", help="신뢰도가 낮아도 보정 파일을 만듦")
    parser.add_argument("--no-ratio", action="store_true", help="프레임레이트 비율 보정을 시도하지 않음")
    parser.add_argument("--max-offset", type=int, default=120, help="찾을 최대 어긋남(초, 기본 120)")
    parser.add_argument("--segment", type=int, default=120, help="구간 점검 단위(초, 기본 120)")
    parser.add_argument("--segment-search", type=int, default=150,
                        help="구간별로 전체 결과에서 더 찾아볼 범위(초, 기본 150)")
    parser.add_argument("--report", default="subtitle_sync_report.csv", help="보고서 파일 이름")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit("폴더를 찾을 수 없습니다: %s" % folder)

    pairs, unmatched = find_pairs(folder, args.recursive, args.suffix)
    print("영상/자막 쌍 %d개, 짝 없는 자막 %d개" % (len(pairs), len(unmatched)))

    rows = []
    for i, (video, sub) in enumerate(pairs, 1):
        print("[%d/%d] %s" % (i, len(pairs), sub.name), flush=True)
        try:
            row = process(video, sub, args)
        except Exception as exc:  # 한 파일 실패가 전체를 멈추지 않도록
            row = {"영상": video.name, "자막": sub.name, "판정": "오류", "메모": str(exc)}
        print("    -> %s  오프셋 %s초  비율 %s  PSR %s  %s" % (
            row.get("판정", ""), row.get("오프셋(초)", ""), row.get("비율", ""),
            row.get("신뢰도(PSR)", ""), row.get("메모", "")))
        rows.append(row)
    for sub in unmatched:
        rows.append({"자막": sub.name, "판정": "짝 없음", "메모": "같은 이름의 영상 파일이 없음"})

    fields = ["영상", "자막", "판정", "오프셋(초)", "비율", "신뢰도(PSR)",
              "구간별 오프셋(초)", "결과 파일", "메모"]
    report = folder / args.report
    with open(report, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {}
    for row in rows:
        summary[row["판정"]] = summary.get(row["판정"], 0) + 1
    print("\n요약: " + ", ".join("%s %d" % (k, v) for k, v in summary.items()))
    print("보고서: %s" % report)


if __name__ == "__main__":
    main()
