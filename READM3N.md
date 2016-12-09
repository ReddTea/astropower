# astropower

Sloan 100000 imagenes

https://arxiv.org/pdf/1503.07077v1.pdf
https://www.youtube.com/watch?v=6eBpjEdgSm0

http://www.sdss.org/dr13/imaging/images/#atlas
https://keras.io/


redshift
petrosian radius & magnitude
r-band (only one)
brightness
magnitude

gal zoo probs % -> tablas ->  eliptica, espiral vs basura
fits atlas, bajar imagenes

JUEVES_1 -> Imagen armada
VIERNES -> 1e3 imagenes descargadas

30 seconds to keras (?) instalar (????)

____________________________________________



http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/[SUBDIR]/atlases/[PID]/[IAUNAME]-[PID]-atlas-[AID].fits.gz (child image)

http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/atlases/0/[IAUNAME]-0-atlas-[AID].fits.gz

read_atlas_image -c 2 fpAtlas-000752-3-0177.fit 432  myAtlasImage.fits

Naming convention: fpAtlas-rrrrrr-c-ffff.fit, where rrrrrr is the imaging run number, c is the column location in the Imaging Array of the CCD which acquired the data (1-6), and ffff is the field number within the run.
______________________________________________________

SELECT TOP 10
  o.objID, o.ra, o.dec, o.run, o.rerun,o.camcol,o.field, o.obj, o.petroMag_r, o.petroRad_r, so.z
  from Galaxy as o, SpecPhoto as so
  WHERE so.objID = o.objID
