# X-OPSTEP
Xinglong-Observatory Popular Science TElescope Pipeline

This pipeline was originally designed for popular science telescopes setted in Xinglong Observatory, NAOC, China.\
But if you change the fits header and filename with the same format as the format X-OPSTEP
need, then observed data from any telescope could be processed.\
X-OPSTEP's goal is to process astronomical images from raw data to PDFs that using a nearby galaxies catalog. With image pre-processing, astrometry, photometry, catalog cross match, flux calibration, image combine and subtract, we finally check whether there are real transients in the observed images. In addition to our scientific objectives, the intermediates generated by the program are also useful, for example, you can just use the WCS catalog to make light curve in order to find the variable stars in images.
