#!/bin/zsh

BIN=../build/motion_blur

if [[ $# != 3 ]]; then
  echo -en "Usage: generate_blog.sh <image> <dest_dir> <html_rel_dir>\n"
  exit
fi

IMG=$1
DEST=$2
HTML_REL_DIR=$3

EXPOSURE_MUL=3

TEMPLATE1='
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3x3 Image Grid</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body, html {
            height: 100%;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 15px; /* Horizontal and vertical spacing */
            height: 100vh;
            width: 80vw;
            padding: 20px;
        }
        .grid-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        img {
            width: 100%;
            height: auto;
        }
        figcaption {
            margin-top: 10px;
            font-size: 1.2em;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="grid-container">
'

TEMPLATE2='
    </div>
</body>
</html>
'

TX=60
TY=15
SC=1.05
ROT=1.5
PR=3e-05

opstrings=(
  "-m Original"
  "-m 'Translation' -sb 20 -tx -$TX $TX -ty -$TY $TY -em $EXPOSURE_MUL"
  "-m 'Translation Wiggle' -sb 20 -tx -$TX $TX -ty -$TY $TY -em $EXPOSURE_MUL -wig"
  "-m 'Rotation' -ref 3000 2000 -r -$ROT $ROT -em $EXPOSURE_MUL"
  "-m 'Scale and Rotation' -sb 30 -ref 3000 2000 -sx 1 $SC -sy 1 $SC -r -$ROT $ROT -em $EXPOSURE_MUL"
  "-m 'Skew' -ref 1350 2000 -sk -0.03 0.03 -em $EXPOSURE_MUL"
  "-m 'Scale' -sb 30 -ref 3000 2000 -sx 1 $SC -sy 1 $SC -em $EXPOSURE_MUL"
  "-m 'Projection' -sb 5 -ref 2000 2000 -px 0 $PR -py 0 -$PR -em $EXPOSURE_MUL"
  "-m 'Translation, Rotation' -sb 20 -tx -$TX $TX -ty -$TY $TY -r -$ROT $ROT -em $EXPOSURE_MUL"
  "-m 'Trans., Scale, Rotation, Projection' -sb 5 -ref 2000 2000 -tx -$TX $TX -ty -$TY $TY -sx 1 $SC -sy 1 $SC -r -$ROT $ROT -px 0 $PR -py 0 -$PR -em $EXPOSURE_MUL"
)

echo "$TEMPLATE1"

IFS=$'\t'
for opts in $opstrings; do
  output=($(eval $BIN $IMG $DEST $opts))
  filepath=${output[1]}
  filename=$(basename $filepath)
  filename_small=${filename/.png/_small.png}
  filepath_small=${filepath/.png/_small.png}
  message=${output[2]}
  convert -resize 25% $filepath $filepath_small

  cat<<EOF
  <figure class="grid-item">
     <a href="$HTML_REL_DIR/$filename">
       <img src="$HTML_REL_DIR/$filename_small" alt="$message" style="width:400px;height:auto;">
    </a>
     <figcaption>$message</figcaptionn>
  </figure>
EOF
done

echo "$TEMPLATE2"



