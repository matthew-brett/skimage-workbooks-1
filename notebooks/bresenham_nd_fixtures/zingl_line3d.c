/* plotLine3d is Alois Zingl's published 3-D Bresenham algorithm, copied
 * verbatim from http://members.chello.at/~easyfilter/bresenham.c
 * (Copyright Alois Zingl, Vienna, Austria; retrieved 2026-09-15).
 * No license is published for that source; see /copyright at the repo
 * root — this file is excluded from the repo's CC0-1.0 dedication.
 * setPixel and main are a CLI wrapper, not part of the original source:
 * `zingl_line3d x0 y0 z0 x1 y1 z1` prints one "x,y,z" triple per rasterised
 * point, so a caller can diff it against another implementation without
 * linking against this file directly.
 */
#include <stdio.h>
#include <stdlib.h>

static void setPixel(int x, int y, int z) {
   printf("%d,%d,%d\n", x, y, z);
}

void plotLine3d(int x0, int y0, int z0, int x1, int y1, int z1)
{
    int dx = abs(x1-x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1-y0), sy = y0 < y1 ? 1 : -1;
    int dz = abs(z1-z0), sz = z0 < z1 ? 1 : -1;
    int dm = dx > dy && dx > dz ? dx : dy > dz ? dy : dz, i = dm; /* max diff */
    x1 = y1 = z1 = dm/2;                                      /* error offset */

    for (;;) {
        setPixel(x0,y0,z0);
        if (i-- == 0) break;
        x1 -= dx; if (x1 < 0) { x1 += dm; x0 += sx; }
        y1 -= dy; if (y1 < 0) { y1 += dm; y0 += sy; }
        z1 -= dz; if (z1 < 0) { z1 += dm; z0 += sz; }
    }
}

int main(int argc, char **argv) {
   if (argc != 7) {
      fprintf(stderr, "usage: %s x0 y0 z0 x1 y1 z1\n", argv[0]);
      return 1;
   }
   plotLine3d(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
              atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
   return 0;
}
