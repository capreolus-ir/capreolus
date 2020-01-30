from django.forms import forms, IntegerField, CharField, BooleanField, FloatField


class QueryForm(forms.Form):
    """
    We don't really need a form, but if we use one django will do all the grunt work of converting strings to numbers
    and booleans e.t.c
    """

    FORM_CLASSES = {}

    target_index = CharField(required=True)
    dataparallel = CharField(required=True)
    batch = IntegerField(required=True)
    benchmark = CharField(required=True)
    collection = CharField(required=True)
    expid = CharField(required=True)
    fold = CharField(required=True)
    index = CharField(required=True)
    indexstops = BooleanField(required=False)
    itersize = IntegerField(required=True)
    lr = FloatField(required=True)
    maxdoclen = IntegerField(required=True)
    maxqlen = IntegerField(required=True)
    reranker = CharField(required=True)
    niters = IntegerField(required=True)
    predontrain = BooleanField(required=False)
    searcher = CharField(required=True)
    rundocsonly = BooleanField(required=False)
    sample = CharField(required=True)
    seed = IntegerField(required=True)
    softmaxloss = BooleanField(required=False)
    stemmer = CharField(required=True)
    query = CharField(required=True)
    gradacc = IntegerField(required=True)

    @classmethod
    def register(cls, formcls):
        name = formcls.name

        if name in cls.FORM_CLASSES and cls.FORM_CLASSES[name] != formcls:
            raise RuntimeError(f"encountered two Forms with the same name: {name}")

        cls.FORM_CLASSES[name] = formcls
        return formcls


@QueryForm.register
class KNRMForm(QueryForm):
    name = "KNRM"
    gradkernels = BooleanField(required=False)
    scoretanh = BooleanField(required=False)
    singlefc = BooleanField(required=False)
    embeddings = CharField(required=True)
    keepstops = BooleanField(required=False)


@QueryForm.register
class PACRRForm(QueryForm):
    name = "PACRR"
    mingram = IntegerField(required=True)
    maxgram = IntegerField(required=True)
    nfilters = IntegerField(required=True)
    idf = BooleanField(required=False)
    kmax = IntegerField(required=True)
    shuf = BooleanField(required=False)
    combine = CharField(required=True)
    rbfkernels = IntegerField(required=True)
    disamb = BooleanField(required=False)
    filtercombine = BooleanField(required=False)
    kernelsimmat = BooleanField(required=False)
    gradkernels = BooleanField(required=False)


@QueryForm.register
class ConvKNRMForm(QueryForm):
    name = "ConvKNRM"
    gradkernels = BooleanField(required=False)
    maxngram = IntegerField(required=True)
    crossmatch = BooleanField(required=False)
    filters = IntegerField(required=True)
    scoretanh = BooleanField(required=False)
    singlefc = BooleanField(required=False)
    embeddings = CharField(required=True)
    keepstops = BooleanField(required=False)


@QueryForm.register
class DRMMForm(QueryForm):
    name = "DRMM"
    nbins = IntegerField(required=True)
    nodes = IntegerField(required=True)
    histType = CharField(required=True)
    gateType = CharField(required=True)


class BM25Form(forms.Form):
    name = "BM25"
    query = CharField(required=True)
    target_index = CharField(required=True)
    reranker = CharField(required=True)
    b = FloatField(required=False)
    k1 = FloatField(required=False)
