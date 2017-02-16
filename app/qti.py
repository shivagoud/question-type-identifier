from nltk.classify.naivebayes import NaiveBayesClassifier
import os

def extract_features(sentence):
  words = sentence.lower().split()
  featureset = dict([('contains-word(%s)' % w, True) for w in words])
  #featureset['contains-phrase(%s %s)'% (words[0],words[1])] = True
  featureset['first-word(%s)'%words[0]] = True    # improvement from 5.33 to 2.76
  #featureset['last-word(%s)'%words[-1]] = True   # no improvement
  return featureset

def train(filename):

  print 'Reading data from the file ' + filename
  labeled_featuresets = []
  with open(filename) as f:
    for line in f:
        sentence, category = line.split(' ,,, ',  1)
        labeled_featuresets.append((extract_features(sentence), category.strip()))

  print 'Training started'
  classifier = NaiveBayesClassifier.train(labeled_featuresets)

  print 'Training completed\n'
  return classifier

def print_classified_probs(prbs):
  print dict([(s, round(100*prbs.prob(s), 2)) for s in prbs.samples()])

def main():
  training_file = os.path.dirname(__file__) + '/data/labelled_data.txt'
  classifier = train(training_file)

  sentence = raw_input("Enter test string: ")
  featurevector = extract_features(sentence)

  #print featurevector

  question_types = classifier.prob_classify(featurevector)

  print_classified_probs(question_types)

  print '\nQuestion Type: '+classifier.classify(featurevector)+'\n\n'


def test():
  training_file = os.path.dirname(__file__) + '/data/labelled_data.txt'
  classifier = train(training_file)

  with open(training_file) as f:
    total=0
    errors=0
    for line in f:
      sentence, category = line.split(' ,,, ',  1)
      classified_category = classifier.classify(extract_features(sentence))
      labelled_category = category.strip()

      if labelled_category != classified_category:
        print 'Sentence: '+sentence
        print 'Labelled: '+labelled_category + '\tClassified: '+classified_category
        errors += 1
      total += 1
    print 'Total error = %.2f %%' % (errors*100.0/total)

if __name__ == '__main__':
  main()
  #test()

