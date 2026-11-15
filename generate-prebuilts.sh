#!/bin/sh
# Regenerate the files Soong cannot produce (meson custom_targets / configure_file).
# Run from the virglrenderer source root; needs python3 with PyYAML.
set -e
export PYTHONDONTWRITEBYTECODE=1
U=src/gallium/auxiliary/util
python3 $U/u_format_table.py $U/u_format.yaml > prebuilt-intermediates/src/u_format_table.c
python3 $U/u_format_table.py $U/u_format.yaml --enums > prebuilt-intermediates/src/u_format_gen.h
python3 src/venus/vkr_device_object.py -o prebuilt-intermediates/venus src/venus/vkr_device_object.json
sed -e "s/@VIRGL_MAJOR_VERSION@/1/;s/@VIRGL_MINOR_VERSION@/3/;s/@VIRGL_MICRO_VERSION@/0/" \
  src/virgl-version.h.meson > prebuilt-intermediates/src/virgl-version.h
