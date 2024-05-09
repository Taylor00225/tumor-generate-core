@echo off
setLocal enableExtensions

REM Examination before starting
if not defined DemoPath (
    echo demo.bat: ERROR:DemoPath is not defined
    goto:EOF
)
echo DemoPath=%DemoPath%
if exist %DemoPath%\\data (
    if not exist %DemoPath%\\data\\imagesTr (
        echo demo.bat: ERROR:%DemoPath%\\data\\imagesTr is not existed
        goto:EOF
    )
    if not exist %DemoPath%\\data\\imagesTs (
        echo demo.bat: ERROR:%DemoPath%\\data\\imagesTs is not existed
        goto:EOF
    )
    if not exist %DemoPath%\\data\\labelsTr (
        echo demo.bat: ERROR:%DemoPath%\\data\\labelsTr is not existed
        goto:EOF
    )
)

REM Be careful!
if not exist %DemoPath%\\output (
    md %DemoPath%\\output
) else (
    rmdir /s /q %DemoPath%\\output
    md %DemoPath%\\output
)
if not exist %DemoPath%\\output_dcm (
    md %DemoPath%\\output_dcm
) else (
    rmdir /s /q %DemoPath%\\output_dcm
    md %DemoPath%\\output_dcm
)

REM Firstly create data json
echo demo.bat: Create data json

if not exist %DemoPath%\\create_data_json.py (
    echo ERROR:Missing create_data_json.py
    goto:EOF
)
python %DemoPath%\\create_data_json.py
echo demo.bat: Json created

REM Secondly start demo
echo demo.bat: Start generate tumor
if not exist %DemoPath%\\generate.py (
    echo demo.bat: ERROR:Missing generate.py
    goto:EOF
)
python %DemoPath%\\generate.py
echo demo.bat: Generation complete

REM Thirdly turn nifti into dicom
python %DemoPath%\\nii2dcm.py
echo demo.bat: Done!