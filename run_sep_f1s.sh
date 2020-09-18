### run these manually!
#before running these, run one item of each domain, to get the indexed cached.
./run_slurm_cache_collections.sh
sleep 120;
./run_BM25_slurm_annotated_pipeline_fold1.sh noneE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/BM25_noneE_f1 ;
sleep 10;
./run_BM25_slurm_annotated_pipeline_fold1.sh allE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/BM25_allE_f1 ;
#after this, we have to wait for it to finish and then run domainE
sleep 10000;
./run_BM25_slurm_annotated_pipeline_fold1.sh domainE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/BM25_domainE_f1 ;
#again we have to wait fot it to finish then run specE
sleep 10000;
./run_BM25_slurm_annotated_pipeline_fold1.sh specE &> /GW/D5data-13/ghazaleh/ranking_outputs/logs_18092020_slurm_jobs/BM25_specE_f1 ;
