Get-ChildItem *.zip | % {& "C:\Program Files\7-Zip\7z.exe" "x" $_.fullname "-oC:\unzipplayground"}
Get-ChildItem *.7z | % {& "C:\Program Files\7-Zip\7z.exe" "x" $_.fullname "-oC:\unzipplayground"}