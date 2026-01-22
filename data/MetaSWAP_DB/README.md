LHM2018_v02v contains results from SWAP simulations, useful for MetaSWAP (commonly used in combinatio with MF in imod_coupler).
The output consists of multiple very small files, so it's advised to compress (could be zip, or 7z, or .tar.gz, depends on the use) before copying somewhere else.
I'll use the command below from the parent folder to zip it:
tar -cf - MetaSWAP_DB | pv | gzip -1 > MSW_DB.tar.gz
