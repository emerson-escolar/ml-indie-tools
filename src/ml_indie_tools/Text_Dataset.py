import random
import logging
try:
   from IPython.core.display import display, HTML
except:
    pass

class Text_Dataset:
    """ Initialize the Text_Dataset with a list of texts.

    The Gutenberg_Dataset can be used to create such a list, by:
        
    .. code-block:: python

        from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
        gd = Gutenberg_Dataset()
        gd.load_index()
        ls = gd.search({'author': 'kant', 'title': 'kritik', 'language': 'german'})  # returns a list of texts
        ls = gd.insert_texts(ls)  # this inserts the actual text of the books into field 'text'.
        # Now ls contains a valid list of text records:
        td = Text_Dataset(ls)
    
    :param text_list: list of text-records of the form: {'author': 'author', 'title': 'title', 'language': 'some-language', 'text': 'the-long-text'}. Optinal parameters: 'weight': 1.0
    """
    def __init__(self, text_list):
        self.log = logging.getLogger("Datasets")
        self.text_list = []
        self.index = 1
        req_attrs=['title', 'author', 'language', 'text']
        for ind in range(0,len(text_list)):
            valid=True
            miss=[]
            for attr in req_attrs:
                if attr not in text_list[ind]:
                    valid=False
                    miss.append(attr)
            if valid is False:
                self.log.error(f"Missing attribute(s) {miss} in text[{ind}], skipping")
                continue
            text=text_list[ind]
            text['index']=self.index
            self.index += 1
            self.text_list.append(text)
        self.log.info(f"Loaded {len(self.text_list)} texts")
        self._calc_probability_weights()

    def _calc_probability_weights(self):
        prs = 0
        for text in self.text_list:
            if 'weight' in text:
                w=text['weight']
            else:
                w=1.0
            pr = len(text['text']) * w
            prs = prs + pr
            text['probability_weight'] = pr
        for text in self.text_list:
            text['probability_weight'] = text['probability_weight'] / prs
        self.tidx = []
        self.tcum = []
        tc=0
        for idx in range(0,len(self.text_list)):
            self.tidx.append(idx)
            text = self.text_list[idx]
            self.tcum.append(text['probability_weight'] + tc)
            tc = self.tcum[-1]

    def _get_random_text_index(self, weighted=True):
        """ Return a random text index from the Text_Dataset.
        
        :param weighted: If True, the probability of a text is weighted by its calculated 'probability_weight' attribute.
        :return: a random text index
        """
        if weighted is True:
            return random.choices(self.tidx, self.tcum)[0]
        else:
            return random.choice(self.tidx)

    def get_random_sample(self, length, weighted=True, sanitize_white_space=True):
        """ Return index idx and random sample of `length` chars from text[idx] the Text_Dataset.
        
        :param length: number of characters to return
        :param weighted: If True, the probability of a text is weighted by its calculated 'probability_weight' attribute.
        :param sanitize_white_space: If True, white space is replaced by a single space.
        :return: tuple (idx of text used for sampling, string of length `length` sampled from the Text_Dataset)
        """
        idx = self._get_random_text_index(weighted)
        text = self.text_list[idx]['text']
        if len(text) < length:
            sample = text
        else:
            pos = random.randint(0, len(text) - length)
            sample = text[pos:pos+length]
        if sanitize_white_space is True:
            sample = sample.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('  ', ' ')
        return (idx, sample)

    def _display_colored_html(self, textlist, dark_mode=False, display_ref_anchor=True, pre='', post=''):
        """ Internal function to display text and citation references in HTML. """
        bgcolorsWht = ['#d4e6e1', '#d8daef', '#ebdef0', '#eadbd8', '#e2d7d5', '#edebd0',
                    '#ecf3cf', '#d4efdf', '#d0ece7', '#d6eaf8', '#d4e6f1', '#d6dbdf',
                    '#f6ddcc', '#fae5d3', '#fdebd0', '#e5e8e8', '#eaeded', '#A9CCE3']
        bgcolorsDrk = ['#342621','#483a2f', '#3b4e20', '#2a3b48', '#324745', '#3d3b30',
                    '#3c235f', '#443f4f', '#403c37', '#463a28', '#443621', '#364b5f',
                    '#264d4c', '#2a3553', '#3d2b40', '#354838', '#3a3d4d', '#594C23']
        if dark_mode is False:
            bgcolors=bgcolorsWht
        else:
            bgcolors=bgcolorsDrk
        out = ''
        for txt, ind in textlist:
            txt = txt.replace('\n', '<br>')
            if ind == 0:
                out += txt
            else:
                if display_ref_anchor is True:
                    anchor="<sup>[" + str(ind) + "]</sup>"
                else:
                    anchor=""
                out += "<span style=\"background-color:"+bgcolors[ind % 16]+";\">" + \
                       txt + "</span>"+ anchor
        display(HTML(pre+out+post))

    def source_highlight(self, ref_txt, minQuoteSize=10, dark_mode=False, display_ref_anchor=True):
        """ Analyse which parts of `ref_txt` are cited from the texts in the Text_Dataset.
        
        Note: this function requires a jupyter notebook in order to display HTML with markup.
        
        :param ref_txt: the reference text to be analysed for plagiarised parts
        :param minQuoteSize: minimum size of a quote to be considered plagiarised
        :param dark_mode: if True, the background colors will be dark, otherwise white
        :param display_ref_anchor: if True, the reference text will be displayed with a reference anchor
        """
        ref_tx = ref_txt
        out = []
        qts = []
        txsrc = [("Sources: ", 0)]
        sc = False
        noquote = ''
        while len(ref_tx) > 0:  # search all library files for quote 'txt'
            mxQ = 0
            mxI = 0
            mxN = ''
            found = False
            for text in self.text_list:  # find longest quote in all texts
                p = minQuoteSize
                if p <= len(ref_tx) and ref_tx[:p] in text['text']:
                    p = minQuoteSize + 1
                    while p <= len(ref_tx) and ref_tx[:p] in text['text']:
                        p += 1
                    if p-1 > mxQ:
                        mxQ = p-1
                        mxI = text['index']
                        mxN = f"{text['author']}: {text['title']}"
                        found = True
            if found:  # save longest quote for colorizing
                if len(noquote) > 0:
                    out.append((noquote, 0))
                    noquote = ''
                out.append((ref_tx[:mxQ], mxI))
                ref_tx = ref_tx[mxQ:]
                if mxI not in qts:  # create a new reference, if first occurence
                    qts.append(mxI)
                    if sc:
                        txsrc.append((", ", 0))
                    sc = True
                    txsrc.append((mxN, mxI))
            else:
                noquote += ref_tx[0]
                ref_tx = ref_tx[1:]
        if len(noquote) > 0:
            out.append((noquote, 0))
            noquote = ''
        self._display_colored_html(out, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor)
        if len(qts) > 0:  # print references, if there is at least one source
            self._display_colored_html(txsrc, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor, pre="<small><p style=\"text-align:right;\">",
                                     post="</p></small>")
