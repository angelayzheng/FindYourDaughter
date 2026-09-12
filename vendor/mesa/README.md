# Bundled Windows software OpenGL

`win_amd64/osmesa.dll` and `win_amd64/libglapi.dll` come from the x64 directory
of the [Mesa3D 24.3.4 MSVC release](https://github.com/pal1000/mesa-dist-win/releases/tag/24.3.4).
The hashes and original archive URL are recorded in `manifest.json`.
The dashboard sets `VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow` and
`GALLIUM_DRIVER=softpipe` only for its VTK subprocess. It does not modify system
drivers or environment settings.

Mesa's component licenses and copyright information are documented by the
[Mesa project](https://docs.mesa3d.org/license.html). The Mesa Windows
distribution is maintained by [pal1000](https://github.com/pal1000/mesa-dist-win).
