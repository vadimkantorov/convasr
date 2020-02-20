IN=https://echo.msk.ru/programs/personalno/
wget --recursive --no-parent -A /archive/,*-echo -A index.html -R comments.html,q.html $IN
