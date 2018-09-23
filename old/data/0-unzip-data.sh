#/bin/sh
$ for file in ls *.zip; do unzip $file -d echo $file | cut -d . -f 1; done
$ for file in ls *.7z; do unzip $file -d echo $file | cut -d . -f 1; done