set -e

PROG=victory

#IN=https://echo.msk.ru/programs/$PROG
#wget --recursive --no-parent -A /archive/,*-echo -A index.html -R comments.html,q.html $IN

#wget --quiet --recursive --no-parent -I /programs/$PROG/archive/ echo.msk.ru/programs/$PROG/index.html
for p in echo.msk.ru/programs/$PROG/index.html echo.msk.ru/programs/$PROG/archive/*/index.html; do
	grep -Po '(?<=href=")/programs/'$PROG'.+-echo' $p
done | sort | uniq | sed 's/^/http:\/\/echo.msk.ru/' > $PROG.txt
