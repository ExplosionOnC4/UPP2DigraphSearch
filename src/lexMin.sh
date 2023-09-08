#!/bin/sh

gap -r -b -q lexMinGAP.g << EOI
lexMin($1, $2);
quit;
EOI