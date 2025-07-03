#!/bin/bash

#langs_pro = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh']
#langs_com = ['cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms']
#langs_mtr = ['el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']

for lang in ar bn de es fr hi id it ja ko pt sw yo zh \
	cs fa tr ro si am te uk vi ru ms \
	el fil ha he ig ky lt mg ne nl ny pl sn so sr sv
do
	python embed_GMMLU.py --lang $lang
done
