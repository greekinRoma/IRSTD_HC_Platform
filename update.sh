unset proxy_https 
unset proxy_http
unset all_proxy
git add *
git commit -m update_files
git push origin master
echo "Files updated and pushed to remote repository."