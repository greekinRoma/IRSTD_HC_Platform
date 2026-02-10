unset all_proxy
unset http_proxy
unset ftp_proxy
unset rsync_proxy
unset https_proxy
if [ -d "data/IRSTD-1K" ] && [ -d "data/NUDT-SIRST" ]; then
    echo "Both directories exist."
else
    echo "At least one directory does not exist."

fi