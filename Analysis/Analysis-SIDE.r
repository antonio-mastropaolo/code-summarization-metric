setwd("summarization-metric")
t<-read.csv("human-annotated-dataset-with-metrics.csv")

summary(t)

newHeader=c("X","user_id", "question_id", "mid", "Overall DA Score", "Content Adequacy",
       	"Conciseness", "Fluency", "sourcerer_id", "codeComment",
       	"codeFunctions", "originalComment", "Jaccard","BLEU-A",
       	"BLEU-1","BLEU-2","BLEU-3","BLEU-4","TF_IDF_CS","USE_CS",
		"BERTScore-P","BERTScore-R","BERTScore-F1","SentenceBERT_CS",
		"InferSent_CS","METEOR","ROUGE-L-F1", "tokenized_function", "c_coeff", "lenComment","num",
		"ROUGE-1-F1","ROUGE-1-P","ROUGE-1-R","ROUGE-2-F1","ROUGE-2-P","ROUGE-2-R","ROUGE-3-F1","ROUGE-3-P","ROUGE-3-R",
       	"ROUGE-4-F1","ROUGE-4-P","ROUGE-4-R","ROUGE-L-P","ROUGE-L-R","ROUGE-W-F1","ROUGE-W-P","ROUGE-W-R","TF_IDF_ED","chrF",
		"USE_ED","SentenceBERT_ED","InferSent_ED","CodeT5-plus_CS","SIDE","SIDE Trivial")

for(x in 1:length(t))
{
    colnames(t)[x]<-newHeader[x]
}

summary(t)

userMetrics<-c("Overall DA Score","Content Adequacy","Conciseness","Fluency")
evalMetrics=c("Jaccard","BLEU-A","BLEU-1","BLEU-2","BLEU-3","BLEU-4","TF_IDF_CS","USE_CS",
"BERTScore-P","BERTScore-R","BERTScore-F1","SentenceBERT_CS","InferSent_CS",
"METEOR","ROUGE-L-F1","c_coeff",
"ROUGE-1-F1","ROUGE-1-P","ROUGE-1-R","ROUGE-2-F1","ROUGE-2-P","ROUGE-2-R","ROUGE-3-F1","ROUGE-3-P","ROUGE-3-R","ROUGE-4-F1",
"ROUGE-4-P","ROUGE-4-R","ROUGE-L-P","ROUGE-L-R","ROUGE-W-F1","ROUGE-W-P","ROUGE-W-R","TF_IDF_ED","chrF",
"USE_ED","SentenceBERT_ED","InferSent_ED","CodeT5-plus_CS","SIDE")
allMetrics=c(userMetrics,evalMetrics)
tsum<-subset(t,t$mid!=0)
attach(tsum,warn.conflicts=FALSE)

summary(tsum)

attach(tsum,warn.conflicts=FALSE)

tx<-subset(tsum,select=userMetrics)
cor(tx,method="spearman")

tmet<-subset(tsum,select=evalMetrics)
cor(tmet,method="spearman")

library(Hmisc)
v<-varclus(as.matrix(tmet,similarity="spearman",type="data.matrix"))
plot(v)

quotedEvalMetrics=c()
for(metric in evalMetrics)
{
    quotedEvalMetrics=c(quotedEvalMetrics,paste("`",metric,"`",sep=""))
}

quotedUserMetrics=c()
for(metric in userMetrics)
{
    quotedUserMetrics=c(quotedUserMetrics,paste("`",metric,"`",sep=""))
}

allmetrics=c(quotedUserMetrics,quotedEvalMetrics)

attach(tsum,warn.conflicts=FALSE)
metricsF=paste(quotedEvalMetrics,collapse="+")
fml=paste("~",metricsF,sep="")
m=redun(as.formula(fml),data=tsum,r2=0.8,nk=0)
reducedMetrics=m$In


print(reducedMetrics)

tred=subset(tsum,select=reducedMetrics)
pc=prcomp(tred)
print(pc)

library(xtable)
xtable(pc,digits=2)

summary(pc)
xtable(summary(pc),digits=2)

library(MASS)

quotedReducedMetrics=c()
for(metric in reducedMetrics)
{
    quotedReducedMetrics=c(quotedReducedMetrics,paste("`",metric,"`",sep=""))
}

allmetrics=c(quotedUserMetrics,quotedReducedMetrics)

attach(tsum,warn.conflicts=FALSE)
metricsF=paste(quotedReducedMetrics,collapse="+")
fml=paste("~",metricsF,sep="")


depM="Overall DA Score"
newFormula=paste("`",depM,"`",fml,sep="")
tredDep=tsum
for(metr in reducedMetrics)
{
   tredDep[[metr]]=(tredDep[[metr]]-min(tredDep[[metr]]))*100/(max(tredDep[[metr]])-min(tredDep[[metr]]))
}
tredDep[[depM]]=as.factor(round(tredDep[[depM]]))

mod<-polr(as.formula(newFormula),data=tredDep,Hess=T)
summary(mod)
coeffs <- coef(summary(mod))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
OR<-exp(coeffs[,"Value"])
tab<-cbind(OR=OR,coeffs, "p value" = round(p,3))
rows=length(reducedMetrics)
xtable(tab[1:rows,],digits=4)

depM="Content Adequacy"
newFormula=paste("`",depM,"`",fml,sep="")
tredDep=tsum
for(metr in reducedMetrics)
{
   tredDep[[metr]]=(tredDep[[metr]]-min(tredDep[[metr]]))*5/(max(tredDep[[metr]])-min(tredDep[[metr]]))
}


tredDep[[depM]]=as.factor(round(tredDep[[depM]]))

mod<-polr(as.formula(newFormula),data=tredDep,Hess=T)
summary(mod)
coeffs <- coef(summary(mod))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
OR<-exp(coeffs[,"Value"])
tab<-cbind(OR=OR,coeffs, "p value" = round(p,3))
rows=length(reducedMetrics)
xtable(tab[1:rows,],digits=4)

depM="Conciseness"
newFormula=paste("`",depM,"`",fml,sep="")
tredDep=tsum
for(metr in reducedMetrics)
{
   tredDep[[metr]]=(tredDep[[metr]]-min(tredDep[[metr]]))*5/(max(tredDep[[metr]])-min(tredDep[[metr]]))
}
tredDep[[depM]]=as.factor(round(tredDep[[depM]]))

mod<-polr(as.formula(newFormula),data=tredDep,Hess=T)
summary(mod)
coeffs <- coef(summary(mod))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
OR<-exp(coeffs[,"Value"])
tab<-cbind(OR=OR,coeffs, "p value" = round(p,3))
rows=length(reducedMetrics)
xtable(tab[1:rows,],digits=4)

depM="Fluency"
newFormula=paste("`",depM,"`",fml,sep="")
tredDep=tsum
for(metr in reducedMetrics)
{
   tredDep[[metr]]=(tredDep[[metr]]-min(tredDep[[metr]]))*5/(max(tredDep[[metr]])-min(tredDep[[metr]]))
}
tredDep[[depM]]=as.factor(round(tredDep[[depM]]))

mod<-polr(as.formula(newFormula),data=tredDep,Hess=T)
summary(mod)
coeffs <- coef(summary(mod))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
OR<-exp(coeffs[,"Value"])
tab<-cbind(OR=OR,coeffs, "p value" = round(p,3))
rows=length(reducedMetrics)
xtable(tab[1:rows,],digits=4)

attach(tsum,warn.conflicts=FALSE)
Overall=(`Content Adequacy`+Fluency+Conciseness)/3

cor.test(Overall,`Overall DA Score`,method="spearman")

depM="Overall"
newFormula=paste("`",depM,"`",fml,sep="")
tredDep=tsum
for(metr in reducedMetrics)
{
   tredDep[[metr]]=(tredDep[[metr]]-min(tredDep[[metr]]))*5/(max(tredDep[[metr]])-min(tredDep[[metr]]))
}
tsum=cbind(tsum,Overall)
tredDep[[depM]]=as.factor(round(tredDep[[depM]]))
mod<-polr(as.formula(newFormula),data=tredDep,Hess=T)
summary(mod)
coeffs <- coef(summary(mod))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
OR<-exp(coeffs[,"Value"])
tab<-cbind(OR=OR,coeffs, "p value" = round(p,3))
rows=length(reducedMetrics)
xtable(tab[1:rows,],digits=4)

attach(tsum,warn.conflicts=FALSE)

tx<-subset(tsum,select=allMetrics)
cor(tx,method="spearman")
