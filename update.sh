unset all_proxy
unset http_proxy
unset ftp_proxy
unset rsync_proxy
unset https_proxy
git add *
git commit -m update_files
git push origin master
echo "Files updated and pushed to remote repository."