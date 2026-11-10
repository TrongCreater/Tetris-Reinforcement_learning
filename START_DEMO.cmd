@echo off
setlocal
cd /d "%~dp0"
if not exist "Desktop\publish_v10\TetrisStudioDesktop.exe" (
  echo Desktop game is missing. Rebuild with: dotnet build Desktop\TetrisStudio.csproj -c Release -o Desktop\publish_v10
  pause
  exit /b 1
)
start "" "%~dp0Desktop\publish_v10\TetrisStudioDesktop.exe"
endlocal
