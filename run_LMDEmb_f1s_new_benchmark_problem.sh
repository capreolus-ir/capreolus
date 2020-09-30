#./run_LMDEmb_slurm_annotated_pipeline_fold1.sh noneE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_noneE ;
#./wait.sh;
#./run_LMDEmb_slurm_annotated_pipeline_fold1.sh allE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_allE ;
#./wait.sh
#./run_LMDEmb_slurm_annotated_pipeline_fold1.sh domainE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_domainE ;
#./wait.sh
./run_LMDEmb_slurm_annotated_pipeline_fold1.sh specE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_specE ;
./wait.sh
./run_LMDEmb_slurm_annotated_pipeline_fold1.sh onlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_onlyNE ;
./wait.sh
./run_LMDEmb_slurm_annotated_pipeline_fold1.sh domainOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_domainOnlyNE ;
./wait.sh
./run_LMDEmb_slurm_annotated_pipeline_fold1.sh specOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/LMDEmb_specOnlyNE ;
./wait.sh

