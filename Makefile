.PHONY: tar

tar:
	tar -cvzf submission.tgz --exclude='__pycache__' \
		code/images/ \
		code/plotting/ \
		code/src/ \
		README.md \
		$$(find code -maxdepth 1 -type f)