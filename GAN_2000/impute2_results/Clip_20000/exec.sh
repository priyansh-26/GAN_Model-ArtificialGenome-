./modules/imputation/impute2 \
 -use_prephased_g \
 -m ./GAN_2000/data\imputation\chr22.map \
 -h ./GAN_2000/data\imputation\Clip_refsize_20000.impute2_haps \
 -l ./GAN_2000/data\imputation\chr22_IMPUTE2_format.legend \
 -known_haps_g ./GAN_2000/data\imputation\imputation_target.impute2_haps \
 -int 20052420 20466893 \
 -Ne 20000 \
 -o ./GAN_2000/impute2_results\Clip_20000\result.impute2