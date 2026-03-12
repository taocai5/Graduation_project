@echo off
if not exist "output" mkdir output
g++ src/main.cpp src/csv_reader.cpp src/kmeans.cpp -o main.exe
if %errorlevel% neq 0 (
    echo Compilation failed.
    exit /b %errorlevel%
)
main.exe
   