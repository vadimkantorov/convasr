curl -X POST \
     -H "Authorization: Bearer ${IAM_TOKEN}" \
	      -H "Transfer-Encoding: chunked" \
		       --data-binary "@speech.ogg" \
			        "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?topic=general&folderId=${FOLDER_ID}"
