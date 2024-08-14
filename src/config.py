from transformers import (BertConfig, RobertaConfig, XLNetConfig, AlbertConfig, LongformerConfig,
                          BertTokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer, LongformerTokenizer,
                          DebertaConfig, DebertaTokenizer, MegatronBertConfig, AutoModel, AutoTokenizer, AutoConfig)
from models import (BertForRelationIdentification, RoBERTaForRelationIdentification,
                    XLNetForRelationIdentification, AlbertForRelationIdentification,
                    LongFormerForRelationIdentification, DebertaForRelationIdentification,
                    MegatronForRelationIdentification)


EN1_START = "[s]"
EN1_END = "[e]"

# keep the seq order
SPEC_TAGS = [EN1_START, EN1_END]

MODEL_REQUIRE_SEGMENT_ID = {'bert', 'xlnet', 'albert', 'deberta', 'megatron'}

MODEL_DICT = {
    "bert": (BertForRelationIdentification, BertConfig, BertTokenizer),
    "megatron": (MegatronForRelationIdentification, MegatronBertConfig, BertTokenizer),
    "roberta": (RoBERTaForRelationIdentification, RobertaConfig, RobertaTokenizer),
    "xlnet": (XLNetForRelationIdentification, XLNetConfig, XLNetTokenizer),
    "albert": (AlbertForRelationIdentification, AlbertConfig, AlbertTokenizer),
    "longformer": (LongFormerForRelationIdentification, LongformerConfig, LongformerTokenizer),
    "deberta": (DebertaForRelationIdentification, DebertaConfig, DebertaTokenizer)
}

TOKENIZER_USE_FOUR_SPECIAL_TOKs = {'roberta', 'longformer'}

# change VERSION if any major updates
VERSION = "0.1"
CONFIG_VERSION_NAME = "REModelVersion"

# add new args associated to version
NEW_ARGS = {"use_focal_loss": False,
            "focal_loss_gamma": 2,
            "use_binary_classification_mode": False,
            "balance_sample_weights": False}
