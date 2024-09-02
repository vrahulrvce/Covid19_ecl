IMPORT ML_Core;
IMPORT LearningTrees;

CoronaRec:=  RECORD
        STRING cough;
        STRING fever;
        STRING sore_throat;
        STRING shortness_of_breath;
        STRING head_ache;
        STRING age_60_and_above;
        STRING test_indication;
        STRING corona_result;
    END;

CoronaNewRec:= RECORD
        INTEGER cough;
        INTEGER fever;
        INTEGER sore_throat;
        INTEGER shortness_of_breath;
        INTEGER head_ache;
        INTEGER age_60_and_above;
        INTEGER test_indication;
        INTEGER corona_result;

    END;


CoronaORec:= RECORD
        STRING test_date;
        CoronaRec;
    END;



CoronaDs := PROJECT(DATASET('~corona_tested_individuals_ver_006.csv',
                    CoronaORec,
                    CSV(HEADING(1),
                        SEPARATOR(','),
                        TERMINATOR(['\n','\r\n','\n\r']))),CoronaRec);


// OUTPUT(CoronaDs);


recordCount := COUNT(CoronaDs);
splitRatio := 0.8;

CoronaDsINT := PROJECT ( CoronaDs , TRANSFORM (CoronaNewRec,
       SELF.cough:= (INTEGER) LEFT.cough,
       SELF.fever:= (INTEGER) LEFT.fever,
       SELF.sore_throat:= (INTEGER) LEFT.sore_throat,
       SELF.shortness_of_breath:= (INTEGER) LEFT.shortness_of_breath,
       SELF.head_ache:= (INTEGER) LEFT.head_ache,
       SELF.age_60_and_above:= IF(LEFT.age_60_and_above='Yes',1,0),
       SELF.test_indication:= MAP(LEFT.test_indication='Abroad'=>0,LEFT.test_indication='Contact with confirmed'=>1,2),
       SELF.corona_result:= MAP(LEFT.corona_result='negative'=>0,LEFT.corona_result='positive'=>1,0)));

Shuffler := RECORD
  CoronaNewReC;
  UNSIGNED4 rnd; // A random number
END;

newDs := PROJECT(CoronaDsINT, TRANSFORM(Shuffler, SELF.rnd := RANDOM(), SELF := LEFT));

shuffledDs := SORT(newDs, rnd);

TrainDs := PROJECT(shuffledDs[1..(recordCount * splitRatio)], RECORDOF(CoronaDsINT));
TestDs := PROJECT(shuffledDs[(recordCount*splitRatio + 1)..recordCount], RECORDOF(CoronaDsINT));

// OUTPUT(TrainDs, NAMED('TrainDataset'));
// OUTPUT(TestDs, NAMED('TestDataset'));

ML_Core.AppendSeqID(TrainDs, id, newTrain);
ML_Core.AppendSeqID(TestDs, id, newTest);

// OUTPUT(newTrain, NAMED('TrainDatasetID'));
// OUTPUT(newTest, NAMED('TestDatasetID'));

ML_Core.ToField(newTrain, TrainNF);
ML_Core.ToField(newTest, TestNF);

// OUTPUT(TrainNF, NAMED('TrainNumericField'));
// OUTPUT(TestNF, NAMED('TestNumericField'));

independent_cols := 7;

X_train := TrainNF(number < independent_cols + 1);
y_train := ML_Core.Discretize.ByRounding(PROJECT(TrainNF(number = independent_cols + 1), TRANSFORM(RECORDOF(LEFT), SELF.number := 1, SELF := LEFT)));

X_test := TestNF(number < independent_cols + 1);
y_test := ML_Core.Discretize.ByRounding(PROJECT(TestNF(number = independent_cols + 1), TRANSFORM(RECORDOF(LEFT), SELF.number := 1, SELF := LEFT)));

// OUTPUT(X_test, NAMED('ActualX'));
// OUTPUT(y_test, NAMED('ActualY'));


// //Training

// classifier := LearningTrees.ClassificationForest(numTrees:=1).GetModel(X_train, y_train);
// // OUTPUT(classifier);
// OUTPUT(classifier, ,'~classifier3::out', OVERWRITE); 

//manual input to prediction
manual_X_rec := RECORD
    INTEGER cough;
    INTEGER fever;
    INTEGER sore_throat;
    INTEGER shortness_of_breath;
    INTEGER head_ache;
    INTEGER age_60_and_above;
    INTEGER test_indication;
    INTEGER corona_result;
END;

//MANUAL PREDICTION
MANUAL_DS := DATASET([{1,0,0,0,0,0,0,1}],manual_X_rec);
ML_Core.AppendSeqID(MANUAL_Ds, id, newManual);
ML_Core.ToField(newManual, ManualNF);
Manual_X_test := ManualNF(number < independent_cols + 1);

//prediction and testing
ClassifierDS:= DATASET('~classifier3::out',ML_Core.Types.Layout_Model2,THOR);

predicted := LearningTrees.ClassificationForest().Classify(classifierDS,Manual_X_test);

OUTPUT(predicted, NAMED('PredictedY'));

// cm := ML_Core.Analysis.Classification.ConfusionMatrix(predicted, y_test);

// OUTPUT(cm, NAMED('ConfusionMatrix'));

// accuracy_values := ML_Core.Analysis.CLassification.Accuracy(predicted, y_test);

// OUTPUT(accuracy_values, NAMED('AccuracyValues'));

