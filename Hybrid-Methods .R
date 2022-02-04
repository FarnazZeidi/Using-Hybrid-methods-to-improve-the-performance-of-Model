# By: Farnaz ZEIDI & Lalah AZAR
# Date: 2020  
# Hybrid machine learning techniques in the R  

####################################################
##Veri Seti Okunur
####################################################
#https://www.kaggle.com/uciml/pima-indians-diabetes-database
#Pima Indians Diabetes Database

DataSet <- read.csv( "diabetes.csv",header = TRUE, sep = ",")
View(DataSet)
summary(DataSet)


##########################################################################################
##################################### Veri Hazirlama ##################################### 
##########################################################################################
#sutun isimlerini degistirme
colnames(DataSet) <- c("GS", "Glikoz", "KanBasinci","TDKK","Insulin", "BMI", "DSF", "Yas", "Sonuc")


#Son sütunu faturaya degistirme
DataSet$Sonuc <- as.factor(DataSet$Sonuc)
levels(DataSet$Sonuc) <- c("Saglam", "Hasta")


#"Hasta" kategorisinin tablosu ilk sirada yazabilmesi icin:
DataSet$Sonuc <- relevel(DataSet$Sonuc, ref = "Hasta")
table(DataSet$Sonuc)

####################################################
##k-Nearest Neighbour Imputation (Kayip Degerlerin Tamamlanmasi)
####################################################

# install.packages("VIM")
library(VIM)
DataSet<-kNN(data=DataSet,variable=c(2,3,4,6), k=17)
View(DataSet)
summary(DataSet)

#Fazla sutunlari kaldirma
DataSet<-DataSet[,1:9]


####################################################
##Normalizasyon
####################################################

#install.packages("clusterSim")
library(clusterSim)
library(cluster)

DataSet[,1:8] <- data.Normalization(DataSet [,1:8],type="n4",normalization="column")
summary(DataSet)


####################################################
##K-Means: Outlier Detection (Uc Noktalar Atilmasi)
####################################################

k<-10
set.seed(4)
k_DataSet <- kmeans(DataSet[,1:8], k)

DataSet$KMeanCValue <- k_DataSet$cluster
table(DataSet$KMeanCValue, DataSet$Sonuc)


#Uç noktalar atilmasi (188 satir)

DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==1) | (DataSet$KMeanCValue!=1) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Hasta" & DataSet$KMeanCValue==2) | (DataSet$KMeanCValue!=2) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==3) | (DataSet$KMeanCValue!=3) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==4) | (DataSet$KMeanCValue!=4) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Hasta" & DataSet$KMeanCValue==5) | (DataSet$KMeanCValue!=5) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==6) | (DataSet$KMeanCValue!=6) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==7) | (DataSet$KMeanCValue!=7) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Hasta" & DataSet$KMeanCValue==8) | (DataSet$KMeanCValue!=8) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Saglam" & DataSet$KMeanCValue==9) | (DataSet$KMeanCValue!=9) ),]
DataSet<-DataSet[which((DataSet$Sonuc=="Hasta" & DataSet$KMeanCValue==10) | (DataSet$KMeanCValue!=10) ),]

table(DataSet$KMeanCValue, DataSet$Sonuc)




##########################################################################################
##################################### Siniflandirma ###################################### 
##########################################################################################

#####################################################
# Hold-out (Model Performans Degerlendirme Yontemi)
#####################################################
#install.packages("caret")
library(caret)
set.seed(10)
egitimIndisleri <- createDataPartition(y = DataSet$Sonuc, p = .80, list = FALSE)
egitim <- DataSet[egitimIndisleri,]
test <- DataSet[-egitimIndisleri,]

# Verisetinde hedef nitelige gore egitim ve test veri setinin ayriliminin incelenmesi
table(DataSet$Sonuc)
table(egitim$Sonuc)
table(test$Sonuc)



####################################################
##SVM
####################################################
# install.packages("kernlab")
library(kernlab)
SVM_modeli <- train(y=egitim$Sonuc, x=egitim[,1:8], method = "svmLinear")

SVM_tahminleri <- predict(SVM_modeli, test[,1:8])
(SVM_tablom <- table(SVM_tahminleri, test[[9]], dnn = c("Tahmini Siniflar", "Gercek Siniflar")))

matSVM<-confusionMatrix(data = SVM_tahminleri, reference = test[[9]], mode = "everything" )
matSVM

####################################################
##Basit (Naive) Bayes
####################################################
# install.packages("e1071")
library(e1071)
naiveB_modeli <- naiveBayes(egitim[,1:8], egitim[[9]])
(nb_tahminleri <- predict(naiveB_modeli, test[,1:8]))

(nb_tablom <- table(nb_tahminleri, test[[9]], dnn = c("Tahmini Siniflar", "Gercek Siniflar")))
matnb<-confusionMatrix(data = nb_tahminleri, reference = test[[9]], mode = "everything" )
matnb


####################################################
##C4.5
####################################################
#install.packages("RWeka")
#library(rJava)

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-14.0.1")
library(RWeka)   

C4.5_modeli <- J48(Sonuc ~ ., data=egitim[,1:9])
print(C4.5_modeli)
summary(C4.5_modeli)
plot(C4.5_modeli)
C4.5_tahminleri <- predict(C4.5_modeli, test[, 1:8])
(C4.5_tablom <- table(C4.5_tahminleri, test[[9]], dnn = c("Tahmini Siniflar", "Gercek Siniflar")))
matC4.5<-confusionMatrix(data = C4.5_tahminleri, reference = test[[9]], mode = "everything" )
matC4.5

##########################################################################################
####################################### Kiyaslama  ####################################### 
##########################################################################################


SonucSVM<-c("SVM",round(matSVM$overall[1],digits=3),round(matSVM$byClass[1],digits=3),round(matSVM$byClass[2],digits=3),round(matSVM$byClass[3],digits=3),round(matSVM$byClass[7],digits=3))
names(SonucSVM)<-c("","Dogruluk","Duyarlilik","Belirleyicilik","Kesinlik","F-Ölçütü")



Sonucnb<-c("Naive Bayes",round(matnb$overall[1],digits=3),round(matnb$byClass[1],digits=3),round(matnb$byClass[2],digits=3),round(matnb$byClass[3],digits=3),round(matnb$byClass[7],digits=3))
names(Sonucnb)<-c("","Dogruluk","Duyarlilik","Belirleyicilik","Kesinlik","F-Ölçütü")

SonucC4.5<-c("C4.5",round(matC4.5$overall[1],digits=3),round(matC4.5$byClass[1],digits=3),round(matC4.5$byClass[2],digits=3),round(matC4.5$byClass[3],digits=3),round(matC4.5$byClass[7],digits=3))
names(SonucC4.5)<-c("","Dogruluk","Duyarlilik","Belirleyicilik","Kesinlik","F-Ölçütü")

SonucSVM
Sonucnb
SonucC4.5

# References
#Kadhm, M. S., Ghindawi, I. W., & Mhawi, D. E. (2018). An accurate diabetes prediction system based on K-means clustering and proposed classification approach. International Journal of Applied Engineering Research, 13(6), 4038-4041.
#Campos, G. O., Zimek, A., Sander, J., Campello, R. J., Micenková, B., Schubert, E., et al. (2016). On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Mining and Knowledge Discovery, 30(4), 891-927.
