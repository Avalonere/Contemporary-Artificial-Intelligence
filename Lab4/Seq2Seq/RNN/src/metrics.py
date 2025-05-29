import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class Metrics:
    def __init__(self):
        self.smooth = SmoothingFunction()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def _convert_to_strings(self, token_list):
        """将token ID列表转换为空格分隔的字符串"""
        return ' '.join([str(x) for x in token_list if x != 0])  # 忽略padding token

    def calculate_bleu(self, references, hypotheses):
        """计算BLEU-4分数"""
        # 将数字列表转换为字符串
        ref_strings = [self._convert_to_strings(ref) for ref in references]
        hyp_strings = [self._convert_to_strings(hyp) for hyp in hypotheses]

        # 转换为BLEU所需格式
        references = [[ref.split()] for ref in ref_strings]
        hypotheses = [hyp.split() for hyp in hyp_strings]

        sentence_bl = np.mean([sentence_bleu(ref, hyp, smoothing_function=self.smooth.method7) for ref, hyp in
                               zip(references, hypotheses)])

        return corpus_bleu(references, hypotheses, smoothing_function=self.smooth.method7), corpus_bleu(references,
                                                                                                        hypotheses,
                                                                                                        smoothing_function=self.smooth.method1), sentence_bl

    def calculate_rouge(self, references, hypotheses):
        """计算ROUGE分数"""
        # 将数字列表转换为字符串
        ref_strings = [self._convert_to_strings(ref) for ref in references]
        hyp_strings = [self._convert_to_strings(hyp) for hyp in hypotheses]

        all_scores = {
            'rouge1': {'p': [], 'r': [], 'f': []},
            'rouge2': {'p': [], 'r': [], 'f': []},
            'rougeL': {'p': [], 'r': [], 'f': []}
        }

        for ref, hyp in zip(ref_strings, hyp_strings):
            score = self.rouge_scorer.score(ref, hyp)
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                all_scores[rouge_type]['p'].append(score[rouge_type].precision)
                all_scores[rouge_type]['r'].append(score[rouge_type].recall)
                all_scores[rouge_type]['f'].append(score[rouge_type].fmeasure)

        avg_scores = {}
        detailed_scores = {}

        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[rouge_type] = np.mean(all_scores[rouge_type]['f'])
            detailed_scores[rouge_type] = {
                'precision': np.mean(all_scores[rouge_type]['p']),
                'recall': np.mean(all_scores[rouge_type]['r']),
                'f1': np.mean(all_scores[rouge_type]['f'])
            }

        return {'avg': avg_scores, 'detailed': detailed_scores}

    def calculate_meteor(self, references, hypotheses):
        """计算METEOR分数"""
        # 将数字列表转换为字符串
        ref_strings = [self._convert_to_strings(ref) for ref in references]
        hyp_strings = [self._convert_to_strings(hyp) for hyp in hypotheses]

        scores = []
        for ref, hyp in zip(ref_strings, hyp_strings):
            score = meteor_score([ref.split()], hyp.split())
            scores.append(score)
        return np.mean(scores)

    def compute_metrics(self, references, hypotheses):
        """计算所有指标"""
        metrics = {
            'bleu4': self.calculate_bleu(references, hypotheses),
            'rouge': self.calculate_rouge(references, hypotheses),
            'meteor': self.calculate_meteor(references, hypotheses)
        }
        return metrics


def test_metrics():
    metrics = Metrics()
    refs = [[1, 2, 3, 4], [3, 4, 5, 6]]
    hyps = [[3, 4, 5, 6], [1, 2, 3, 4]]

    results = metrics.compute_metrics(refs, hyps)
    print("测试结果:", results)


if __name__ == "__main__":
    test_metrics()
