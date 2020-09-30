#./run_BM25s.sh ;
#sleep 100;
#./run_LMDs.sh ;
#sleep 100;
./run_BM25_PVs.sh 
sleep 600;
./run_LMD_PVs.sh 
sleep 600;
#sleep 100; run these manually as well:
./run_LMDEmbs.sh
#sleep 100;
#./run_LMDEmb_PVs.sh

