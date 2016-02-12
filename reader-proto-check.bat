@echo off
setlocal
cd %~dp0
for %%f in (%*) do set a_%%f=1

@REM optionally do clean ?

set ACML_FMA=0
set CYGWIN_BIN=c:\cygwin64\bin
if not exist %CYGWIN_BIN% (
    set CYGWIN_BIN=c:\cygwin\bin
    if not exist %CYGWIN_BIN% (
        echo Can't find Cygwin, is it installed?
        exit /b 1
    )
)
echo on

set TEST_SPEC=^
  -t ReaderTestSuite/HTKMLFReaderSimpleDataLoop1 ^
  -t +ReaderTestSuite/HTKMLFReaderSimpleDataLoop4 ^
  -t +ReaderTestSuite/HTKMLFReaderSimpleDataLoop5 ^
  -t +ReaderTestSuite/HTKMLFReaderSimpleDataLoop11

:: NEEDS WORK
::   HTKMLFReaderSimpleDataLoop21_Config.cntk // distributed
::     HTKMLFReaderSimpleDataLoop21_{0,1}
:: 
:: NO
::   UCIFastReaderSimpleDataLoop_Config.cntk // other reader
::   HTKMLFReaderSimpleDataLoop3_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop4_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop10_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop12_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop13_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop16_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop17_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop19_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop2_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop20_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop22_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop6_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop7_Config.cntk // rollingWindow
::   HTKMLFReaderSimpleDataLoop8_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop9_Config.cntk // !frameMode
::   HTKMLFReaderSimpleDataLoop14_Config.cntk // !frameMode

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
if errorlevel 1 exit /b 1

if not defined a_nodebug (
    msbuild /m /p:Platform=x64 /p:Configuration=Debug CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_notests (
if not defined a_nounittests (
        .\x64\Debug\UnitTests\ReaderTests.exe %TEST_SPEC%
        if errorlevel 1 exit /b 1
)
)
)

if not defined a_norelease (
    msbuild /m /p:Platform=x64 /p:Configuration=Release CNTK.sln
    if errorlevel 1 exit /b 1

if not defined a_notests (
if not defined a_nounittests (
    .\x64\Release\UnitTests\ReaderTests.exe %TEST_SPEC%
    if errorlevel 1 exit /b 1
)
)
)

set PATH=%PATH%;%CYGWIN_BIN%

if not defined a_nospeech (
if not defined a_noe2e (
if not defined a_notests (
if not defined a_norelease (
if not defined a_nogpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -d gpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/EndToEndTests/TestDriver.py run -d cpu -f release Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

if not defined a_nodebug (
if not defined a_nogpu (
    python2.7.exe Tests/EndToEndTests/TestDriver.py run -d gpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)

    python2.7.exe Tests/EndToEndTests/TestDriver.py run -d cpu -f debug Speech/QuickE2E
    if errorlevel 1 exit /b 1
)
)
)
)
