#!/bin/bash

#langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh_cn']

for lang in ar da de en es fi fr he hu it ja ko km ms nl no pl pt ru sv th tr vi zh_cn
do
	python embed_MKQA.py --lang $lang
done
