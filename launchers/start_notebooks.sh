set -e
set -u
pushd eli5_retrieval_large_lm/notebooks > /dev/null
screen -S jupyter -dm jupyter lab --ip 0.0.0.0
popd > /dev/null
SLEEP_AMOUNT=0.1

while true ; do	
	TOKEN_STR="$(jupyter lab list)"
	TOKEN_STR="$(echo $TOKEN_STR | grep -oe '\?token=[a-z0-9]\+' || true)" 
	# echo token_str $TOKEN_STR
	if [[ "$TOKEN_STR" != "" ]] ; then
		break
	fi
	sleep "$SLEEP_AMOUNT"
	# echo "Waiting for jupyter."
done
# echo AFTER JUPYTER

echo "$TOKEN_STR"
echo "$TOKEN_STR" > jupyter.txt