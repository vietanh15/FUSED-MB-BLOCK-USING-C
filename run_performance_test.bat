@echo off
REM Compile and run both profiled versions

echo.
echo ========== BUILDING LAYER1_PROFILED.C (Sequential) ==========
gcc -O2 -lm -o layer1_profiled.exe layer1_profiled.c
if %ERRORLEVEL% NEQ 0 (
    echo Build failed for layer1_profiled.c
    exit /b 1
)
echo Build successful!

echo.
echo ========== BUILDING LAYER1WITHPINGPONGMUX_PROFILED.C (Ping-Pong) ==========
gcc -O2 -fopenmp -lm -o layer1_pingpong_profiled.exe layer1withpingpongmux_profiled.c
if %ERRORLEVEL% NEQ 0 (
    echo Build failed for layer1withpingpongmux_profiled.c
    exit /b 1
)
echo Build successful!

echo.
echo ========== RUNNING SEQUENTIAL VERSION ==========
layer1_profiled.exe > result_sequential.txt 2>&1
type result_sequential.txt

echo.
echo ========== RUNNING PING-PONG VERSION ==========
layer1_pingpong_profiled.exe > result_pingpong.txt 2>&1
type result_pingpong.txt

echo.
echo ========== COMPARISON ==========
echo Results saved to:
echo  - result_sequential.txt
echo  - result_pingpong.txt
echo.
pause
