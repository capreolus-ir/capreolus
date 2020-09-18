./run_BM25_slurm_annotated_pipeline_fold1.sh onlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_onlyNE_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_fold1.sh domainOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_domainOnlyNE_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_fold1.sh specOnltNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_specOnlyNE_f1 ;
sleep 10;

./run_BM25_slurm_annotated_pipeline_dv_fold1.sh noneE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_noneE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh allE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_allE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh domainE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_domainE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh specE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_specE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh onlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_onlyNE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh domainOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_domainOnlyNE_dv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_dv_fold1.sh specOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_specOnlyNE_dv_f1 ;
sleep 10;

./run_BM25_slurm_annotated_pipeline_pv_fold1.sh noneE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_noneE_pv_f1
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh allE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_allE_pv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh domainE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_domainE_pv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh specE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_specE_pv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh onlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_onlyNE_pv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh domainOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_domainOnlyNE_pv_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_pv_fold1.sh specOnlyNE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_17092020_slurm_jobs/BM25_specOnlyNE_pv_f1 ;
# then we have to wait for these to finish to run the rest.
