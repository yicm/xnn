#!/bin/bash
# 将所有目录及子目录下的所有文件中含有XNN替换成XNN
grep "XNN" * -R | awk -F: '{print $1}' | sort | uniq | xargs sed -i 's/XNN/XNN/g'
