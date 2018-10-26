#!/bin/bash -eux


if [[ "$OSTYPE" == "darwin"* ]]; then
  RUNTIME=""
  _DISPLAY="$(ifconfig en0 | grep "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | sed -e "s/ netmask .*$//g" | sed -e "s/.*inet //g"):0"
  socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
else
  RUNTIME="--runtime nvidia"
  _DISPLAY=$DISPLAY
fi

docker run -it --rm \
    $RUNTIME \
    --shm-size=256m \
    -u app \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=$_DISPLAY \
    -v ${PWD}:/home/app \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    takuseno/dqn \
    python train.py $*
