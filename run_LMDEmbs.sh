./run_LMDEmb_slurm_annotated_pipeline.sh noneE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_noneE ;
sleep 10;
./run_LMDEmb_slurm_annotated_pipeline.sh allE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_allE ;
sleep 10;
./run_LMDEmb_slurm_annotated_pipeline.sh domainE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_domainE ;
sleep 10;
./run_LMDEmb_slurm_annotated_pipeline.sh specE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_specE ;
sleep 300;
./run_LMDEmb_slurm_annotated_pipeline.sh onlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_onlyNE ;
sleep 10;
./run_LMDEmb_slurm_annotated_pipeline.sh domainOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_domainOnlyNE ;
sleep 10;
./run_LMDEmb_slurm_annotated_pipeline.sh specOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/LMDEmb_specOnlyNE ;
sleep 10;