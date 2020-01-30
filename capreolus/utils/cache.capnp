@0xa7a33d9573f6f32c;

struct Document {
  docid @0 :Text;
  txt @1 :Text;
}

struct EmbedTextInstance {
  query @0 :Data;
  posdoc @1 :Data;
  negdoc @2 :Data;
  queryidf @3 :Data;
}

struct BertTextInstance {
  postoks @0 :Data;
  posmask @1 :Data;
  possegs @2 :Data;
  posqmask @3 :Data;
  posdmask @4 :Data;
  negtoks @5 :Data;
  negmask @6 :Data;
  negsegs @7 :Data;
  negqmask @8 :Data;
  negdmask @9 :Data;
}

struct CacheIdsInstance {
  qid @0 :Text;
  posdocid @1 :Text;
  negdocid @2 :Text;
}

