#!/usr/bin/env python3
"""폴더 안의 영상/자막 쌍을 찾아 싱크를 점검하고 맞추는 스크립트.

영상의 음성에서 대사가 있는 구간을 찾고, 자막이 표시되는 구간과 비교해서
자막을 얼마나 옮겨야 하는지(시간 이동)와 프레임레이트 차이(비율)를 계산한다.

지원 자막: .smi / .srt / .ass / .ssa
필요 프로그램: Python 3.8+, numpy, ffmpeg (PATH에 등록)

처리 순서 (파일마다):
    1) 점검  : 원본 자막이 음성과 맞는지 검증한다. 맞으면 그대로 둔다.
    2) 수정  : 전체 이동 / 프레임레이트 비율 / 구간별 보정 중 간단한 것부터 만든다.
    3) 검증  : 보정한 자막을 음성과 다시 비교한다.
    4) 재수정: 검증에 실패하면 다음 방법으로 다시 시도한다.
    통과한 보정만 원본 자막에 덮어쓴다. 원본은 처음 한 번 "_자막원본백업" 폴더에 보관한다.

사용 예:
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난" --recursive --check
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난" --recursive
    python subtitle_sync.py "F:\\[애니]\\[일본] 명탐정 코난" --restore
"""

import argparse
import bisect
import csv
import re
import shutil
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
PSR_MIN = 5.0
# 이보다 작은 차이는 "정상"으로 본다
OK_OFFSET_MS = 100
# 구간별 편차가 이보다 크면 구간별 보정을 시도한다
SPREAD_WARN_MS = 500
# 영상 전체에 걸쳐 어긋남이 이만큼 이상 변할 때만 프레임레이트 비율 보정을 고려한다
DRIFT_MIN_MS = 400
# 추정한 비율이 흔한 프레임레이트 비율과 이 정도 안으로 일치해야 적용한다
RATIO_TOL = 0.0004
# 이보다 1에 가까운 비율은 "작은 비율"로 보고 구간별 추세로만 판단한다
SMALL_RATIO = 0.005
# 큰 비율은 비율 1.0보다 일치도가 이 배수 이상 높아야 인정한다
BIG_RATIO_GAIN = 1.5
# 검증: 보정 후 다시 찾아볼 범위와 합격 기준
VERIFY_SEARCH_MS = 10000
RESIDUAL_OK_MS = 150
SEG_RESIDUAL_OK_MS = 500
# 보정 후 일치도가 원본보다 최소 이만큼(비율) 올라야 "개선"으로 인정
SCORE_GAIN_MIN = 0.01

# 덮어쓰기 전에 원본 자막을 보관하는 폴더 이름 (자막이 있는 폴더 안에 만들어진다)
BACKUP_DIR = "_자막원본백업"


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


def pad_audio(audio_sig, cues):
    """자막이 영상보다 길게 이어져도 계산할 수 있도록 음성 신호 뒤를 0으로 채운다."""
    length = max(len(audio_sig), int(cues[-1][1] * 1.3 / FRAME_MS) + 1)
    audio = np.zeros(length, dtype=np.float32)
    audio[: len(audio_sig)] = audio_sig
    return audio


def global_search(audio, cues, ratio, max_offset_ms):
    """정해진 비율에서 전체 자막을 가장 잘 맞추는 (오프셋ms, PSR, 일치도)."""
    sub = cue_signal(cues, len(audio), ratio)
    lags, values = correlate(audio, sub, max_offset_ms // FRAME_MS)
    lag, score, psr = peak_info(lags, values)
    return lag * FRAME_MS, psr, score


def reliable_segments(segments):
    result = []
    for seg_start, off, psr, _ in segments:
        if psr >= PSR_MIN:
            result.append((seg_start, off))
    return result


def estimate_ratio(segments):
    """구간별 오프셋이 시간에 따라 일정하게 커지거나 작아지면 프레임레이트 차이로 본다.

    반환: (비율, 영상 전체에 걸친 어긋남 변화량ms)
    잡음으로 비율을 잘못 고르지 않도록, 변화량이 충분히 크고 흔한 프레임레이트
    비율과 거의 일치할 때만 1.0이 아닌 값을 돌려준다.
    """
    points = reliable_segments(segments)
    if len(points) < 3:
        return 1.0, 0.0
    times = np.array([p[0] for p in points], dtype=np.float64)
    offsets = np.array([p[1] for p in points], dtype=np.float64)
    slope = float(np.polyfit(times, offsets, 1)[0])
    drift = slope * (times[-1] - times[0])
    if abs(drift) < DRIFT_MIN_MS:
        return 1.0, drift
    estimate = 1.0 + slope
    for ratio in FPS_RATIOS:
        if ratio != 1.0 and abs(ratio - estimate) <= RATIO_TOL:
            return ratio, drift
    return 1.0, drift


def map_cues(cues, tmap):
    mapped = []
    for start, end in cues:
        new_start = tmap(start)
        mapped.append((new_start, new_start + (end - start) * tmap.ratio))
    mapped.sort()
    return mapped


def evaluate(audio, cues, tmap, segment_ms, search_ms):
    """보정을 적용한 자막이 음성과 얼마나 맞는지 다시 잰다 (검증).

    score    : 그대로 겹쳤을 때의 일치도 (높을수록 좋음)
    residual : 다시 찾아본 전체 어긋남 (0에 가까울수록 좋음)
    seg_max  : 구간별로 다시 찾아본 어긋남 중 가장 큰 값
    """
    mapped = map_cues(cues, tmap)
    sub = cue_signal(mapped, len(audio), 1.0)
    max_lag = VERIFY_SEARCH_MS // FRAME_MS
    lags, values = correlate(audio, sub, max_lag)
    lag, _, _ = peak_info(lags, values)
    # 구간 검증은 넓게 찾아야 광고 컷처럼 크게 어긋난 구간도 잡아낸다
    segments = segment_offsets(audio, mapped, 1.0, 0, segment_ms, search_ms)
    seg_max = 0
    seg_count = 0
    for _, off in reliable_segments(segments):
        seg_count += 1
        seg_max = max(seg_max, abs(off))
    return {
        "score": float(values[max_lag]),
        "residual": lag * FRAME_MS,
        "seg_max": seg_max,
        "seg_count": seg_count,
    }


def passes(ev):
    return abs(ev["residual"]) <= RESIDUAL_OK_MS and ev["seg_max"] <= SEG_RESIDUAL_OK_MS


def describe(ev):
    return "일치도 %.3f, 잔여 %+.2f초, 구간 최대 %.2f초" % (
        ev["score"], ev["residual"] / 1000.0, ev["seg_max"] / 1000.0)


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

def find_pairs(folder, recursive):
    pattern = "**/*" if recursive else "*"
    videos = {}
    subs = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file() or BACKUP_DIR in path.parts:
            continue
        ext = path.suffix.lower()
        if ext in VIDEO_EXTS:
            videos[(path.parent, path.stem.lower())] = path
        elif ext in SUB_EXTS and not path.stem.lower().endswith(".synced"):
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


def backup_path(sub):
    return sub.parent / BACKUP_DIR / sub.name


def build_candidates(audio, cues, args):
    """보정 후보를 간단한 것부터 순서대로 만든다. [(이름, TimeMap, PSR, 구간정보), ...]"""
    max_offset = args.max_offset * 1000
    segment_ms = args.segment * 1000
    search_ms = args.segment_search * 1000

    base_off, base_psr, base_score = global_search(audio, cues, 1.0, max_offset)
    base_segments = segment_offsets(audio, cues, 1.0, base_off, segment_ms, search_ms)

    ratio = 1.0
    drift = 0.0
    if not args.no_ratio:
        # 큰 비율 차이(약 4% 이상)는 구간 안에서도 번져 보이므로 전체 상관으로 찾는다
        best_score = base_score
        for candidate in FPS_RATIOS:
            if abs(candidate - 1.0) < SMALL_RATIO:
                continue
            _, r_psr, r_score = global_search(audio, cues, candidate, max_offset)
            if r_psr >= PSR_MIN and r_score > best_score * BIG_RATIO_GAIN:
                best_score = r_score
                ratio = candidate
        # 작은 비율 차이(23.976 <-> 24)는 잡음에 약하므로 구간별 추세로만 판단한다
        if ratio == 1.0:
            ratio, drift = estimate_ratio(base_segments)

    candidates = []
    split_source = (1.0, base_off, base_psr, base_segments)
    if ratio != 1.0:
        r_off, r_psr, _ = global_search(audio, cues, ratio, max_offset)
        r_segments = segment_offsets(audio, cues, ratio, r_off, segment_ms, search_ms)
        candidates.append(("비율+이동", TimeMap(ratio, [(0, r_off)]), r_psr))
        split_source = (ratio, r_off, r_psr, r_segments)
    candidates.append(("전체 이동", TimeMap(1.0, [(0, base_off)]), base_psr))

    if not args.no_split:
        s_ratio, s_off, s_psr, s_segments = split_source
        points = reliable_segments(s_segments)
        offsets = [o for _, o in points]
        if len(offsets) >= 2 and max(offsets) - min(offsets) > SPREAD_WARN_MS:
            pieces = refine_bounds(audio, cues, s_ratio, piecewise_offsets(s_segments, s_off))
            candidates.append(("구간별", TimeMap(s_ratio, pieces), s_psr))

    info = {"base_off": base_off, "base_psr": base_psr, "drift": drift,
            "segments": base_segments}
    return candidates, info


def process(video, sub, args):
    row = {
        "폴더": "", "자막": sub.name, "판정": "", "적용 방법": "", "오프셋(초)": "",
        "비율": "", "신뢰도(PSR)": "", "원본 검증": "", "보정 후 검증": "",
        "시도 내역": "", "메모": "",
    }
    # 이미 덮어쓴 적이 있으면 백업해 둔 원본을 기준으로 다시 계산한다 (반복 실행해도 안전)
    backup = backup_path(sub)
    source = backup if backup.exists() else sub
    text, encoding = read_text(source)
    ext = sub.suffix.lower()
    cues = parse_cues(text, ext)
    if len(cues) < 10:
        row["판정"] = "건너뜀"
        row["메모"] = "자막 대사를 읽지 못함 (대사 %d개)" % len(cues)
        return row

    audio = pad_audio(speech_signal(extract_audio(video)), cues)
    segment_ms = args.segment * 1000
    search_ms = args.segment_search * 1000
    candidates, info = build_candidates(audio, cues, args)
    row["신뢰도(PSR)"] = "%.1f" % info["base_psr"]

    # 1) 점검: 원본 그대로 검증
    original = evaluate(audio, cues, TimeMap(1.0, [(0, 0)]), segment_ms, search_ms)
    row["원본 검증"] = describe(original)
    if passes(original):
        row["판정"] = "정상"
        row["오프셋(초)"] = "%+.2f" % (original["residual"] / 1000.0)
        if source is backup:
            row["메모"] = "백업 원본 기준으로 정상"
        return row

    # 2) 수정 -> 3) 검증 -> 실패하면 다음 방법으로 재수정
    attempts = []
    chosen = None
    best_any = None
    min_score = original["score"] + SCORE_GAIN_MIN * abs(original["score"])
    for name, tmap, psr in candidates:
        ev = evaluate(audio, cues, tmap, segment_ms, search_ms)
        if best_any is None or ev["score"] > best_any[2]["score"]:
            best_any = (name, tmap, ev, psr)
        if psr < PSR_MIN:
            attempts.append("%s: 신뢰도 낮음(PSR %.1f)" % (name, psr))
            continue
        if not passes(ev):
            attempts.append("%s: 검증 실패(%s)" % (name, describe(ev)))
            continue
        if ev["score"] < min_score:
            attempts.append("%s: 개선 없음(%s)" % (name, describe(ev)))
            continue
        attempts.append("%s: 통과" % name)
        chosen = (name, tmap, ev, psr)
        break
    row["시도 내역"] = " / ".join(attempts)

    if chosen is None:
        if args.force and best_any is not None:
            chosen = best_any
            row["메모"] = "검증 통과 못 했지만 --force로 적용"
        else:
            row["판정"] = "수동 확인 필요"
            row["메모"] = "모든 보정 방법이 검증을 통과하지 못해 원본 유지"
            return row

    name, tmap, ev, psr = chosen
    row["적용 방법"] = name
    row["신뢰도(PSR)"] = "%.1f" % psr
    row["비율"] = "%.5f" % tmap.ratio
    offsets = []
    for off in tmap.offsets:
        offsets.append("%+.2f" % (off / 1000.0))
    row["오프셋(초)"] = " → ".join(offsets)
    row["보정 후 검증"] = describe(ev)

    if args.check:
        row["판정"] = "보정 필요"
        return row

    # 4) 덮어쓰기: 처음 한 번만 원본을 백업해 둔다
    if not backup.exists():
        backup.parent.mkdir(exist_ok=True)
        shutil.copy2(str(sub), str(backup))
    used = write_text(sub, apply_map(text, ext, tmap), encoding)
    row["판정"] = "보정 완료"
    if used != encoding:
        row["메모"] = (row["메모"] + "; " if row["메모"] else "") + "인코딩 %s -> %s" % (encoding, used)
    return row


def restore(folder):
    """백업 폴더의 원본 자막을 원래 위치로 되돌린다."""
    count = 0
    for backup_dir in sorted(folder.glob("**/" + BACKUP_DIR)):
        if not backup_dir.is_dir():
            continue
        for path in sorted(backup_dir.iterdir()):
            if path.is_file():
                shutil.copy2(str(path), str(backup_dir.parent / path.name))
                count += 1
    print("원본 자막 %d개를 되돌렸습니다." % count)


def main(argv=None):
    parser = argparse.ArgumentParser(description="폴더 안의 영상/자막 싱크를 점검하고 맞춥니다.")
    parser.add_argument("folder", help="영상과 자막이 있는 폴더")
    parser.add_argument("--check", action="store_true", help="점검만 하고 자막은 수정하지 않음")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더까지 검사")
    parser.add_argument("--restore", action="store_true", help="백업해 둔 원본 자막으로 되돌림")
    parser.add_argument("--force", action="store_true", help="검증을 통과하지 못해도 가장 나은 보정을 적용")
    parser.add_argument("--no-ratio", action="store_true", help="프레임레이트 비율 보정을 시도하지 않음")
    parser.add_argument("--no-split", action="store_true", help="구간별 보정을 시도하지 않음")
    parser.add_argument("--max-offset", type=int, default=120, help="찾을 최대 어긋남(초, 기본 120)")
    parser.add_argument("--segment", type=int, default=120, help="구간 점검 단위(초, 기본 120)")
    parser.add_argument("--segment-search", type=int, default=150,
                        help="구간별로 전체 결과에서 더 찾아볼 범위(초, 기본 150)")
    parser.add_argument("--report", default="subtitle_sync_report.csv", help="보고서 파일 이름")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit("폴더를 찾을 수 없습니다: %s" % folder)
    if args.restore:
        restore(folder)
        return

    pairs, unmatched = find_pairs(folder, args.recursive)
    print("영상/자막 쌍 %d개, 짝 없는 자막 %d개" % (len(pairs), len(unmatched)))

    rows = []
    for i, (video, sub) in enumerate(pairs, 1):
        print("[%d/%d] %s" % (i, len(pairs), sub.name), flush=True)
        try:
            row = process(video, sub, args)
        except Exception as exc:  # 한 파일 실패가 전체를 멈추지 않도록
            row = {"자막": sub.name, "판정": "오류", "메모": str(exc)}
        row["폴더"] = str(sub.parent.relative_to(folder))
        print("    -> %s  %s  %s" % (row.get("판정", ""), row.get("적용 방법", ""),
                                    row.get("메모", "") or row.get("시도 내역", "")))
        rows.append(row)
    for sub in unmatched:
        rows.append({"폴더": str(sub.parent.relative_to(folder)), "자막": sub.name,
                     "판정": "짝 없음", "메모": "같은 이름의 영상 파일이 없음"})

    fields = ["폴더", "자막", "판정", "적용 방법", "오프셋(초)", "비율", "신뢰도(PSR)",
              "원본 검증", "보정 후 검증", "시도 내역", "메모"]
    report = folder / args.report
    with open(report, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {}
    for row in rows:
        summary[row["판정"]] = summary.get(row["판정"], 0) + 1
    parts = []
    for key, value in summary.items():
        parts.append("%s %d" % (key, value))
    print("\n요약: " + ", ".join(parts))
    print("보고서: %s" % report)


if __name__ == "__main__":
    main()
