BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(32000, 1024, padding_idx=3)
      (position_embeddings): Embedding(512, 1024)
      (token_type_embeddings): Embedding(2, 1024)
      (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-23): 24 x BertLayer(
          (pre_attention_ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (post_attention_ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=1024, out_features=1024, bias=True)
              (key): Linear(in_features=1024, out_features=1024, bias=True)
              (value): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (last_layer_ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=1024, out_features=1024, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
      (decoder): Linear(in_features=1024, out_features=32000, bias=True)
    )
  )
)