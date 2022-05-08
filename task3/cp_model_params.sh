#!/bin/bash

counter=0
for file in $(find . -name "*food_taster.pt")
do
	echo "moving file ${file}, counter = ${counter}"
	# counter=$[counter + 1]
	dir_name=$(dirname ${file})
	desc=$(cat ${dir_name}/desc.*)
	if [[ ${desc} == *101* ]]
	then
		echo "${desc}"
		cp ${file} "trained_params/food_taster_resnet_101_${counter}.pt"

	fi
	echo "$(cat ${dir_name}/desc.*)"
	(( counter++ ))
done

