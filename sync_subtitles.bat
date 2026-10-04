@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem 사용법: 이 파일을 더블클릭하거나, 폴더를 이 파일 위로 끌어다 놓으세요.
rem 끌어다 놓은 폴더가 없으면 아래 기본 폴더를 처리합니다.
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=F:\[애니]\[일본] 명탐정 코난"
set "SCRIPT=%~dp0subtitle_sync.py"

echo ================================================
echo  영상/자막 싱크 일괄 보정
echo  대상 폴더: %TARGET%
echo ================================================

if not exist "%SCRIPT%" goto no_script
if not exist "%TARGET%\" goto no_folder

where python >nul 2>nul
if errorlevel 1 goto no_python

where ffmpeg >nul 2>nul
if errorlevel 1 goto no_ffmpeg

python -c "import numpy" >nul 2>nul
if errorlevel 1 (
    echo numpy를 설치합니다...
    python -m pip install numpy
    if errorlevel 1 goto no_numpy
)

python "%SCRIPT%" "%TARGET%" --recursive
echo.
echo 완료되었습니다. 대상 폴더의 subtitle_sync_report.csv 에서 결과를 확인하세요.
pause
exit /b 0

:no_script
echo [오류] subtitle_sync.py 를 찾을 수 없습니다. 이 bat 파일과 같은 폴더에 두세요.
pause
exit /b 1

:no_folder
echo [오류] 폴더를 찾을 수 없습니다: %TARGET%
pause
exit /b 1

:no_python
echo [오류] Python이 설치되어 있지 않습니다. https://www.python.org 에서 설치하세요.
echo        설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요.
pause
exit /b 1

:no_ffmpeg
echo [오류] ffmpeg를 찾을 수 없습니다. 설치 후 PATH에 등록하세요.
echo        예: 명령 프롬프트에서 winget install ffmpeg
pause
exit /b 1

:no_numpy
echo [오류] numpy 설치에 실패했습니다. 직접 "python -m pip install numpy" 를 실행해 보세요.
pause
exit /b 1
