#--------------- Project-------------------

babies_df <- read.csv("babies.csv")   # read csv file
head(babies_df).  #print top 5 rows 

summary(babies_df)  #gives detailed summary with no. of NA values

#------after finding columns with NA value we will replace NA values with median values of the column-----

#---- columns with NA values are: gestation, age, height, weight, smoke

new_df$gestation  <-ifelse(is.na(new_df$gestation),median(new_df$gestation, na.rm = TRUE),new_df$gestation) 
new_df$age  <-ifelse(is.na(new_df$age),median(new_df$age, na.rm = TRUE),new_df$age)
new_df$height  <-ifelse(is.na(new_df$height),median(new_df$height, na.rm = TRUE),new_df$height)
new_df$weight  <-ifelse(is.na(new_df$weight),median(new_df$weight, na.rm = TRUE),new_df$weight)
new_df$smoke  <-ifelse(is.na(new_df$smoke),median(new_df$smoke, na.rm = TRUE),new_df$smoke)
summary(new_df)   #summary shows no NA values in any column

#------- since dataset is large we will randomly select 1000 rows from dataset

df <- sample_n(new_df,1000)

#--finding correlation----
cor(new_df)   #this give numerical presentaion of correlation

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...) {
  +     usr <- par("usr")
  +     on.exit(par(usr))
  +     par(usr = c(0, 1, 0, 1))
  +     Cor <- abs(cor(x, y)) # Remove abs function if desired
  +     txt <- paste0(prefix, format(c(Cor, 0.123456789), digits = digits)[1])
  +     if(missing(cex.cor)) {
    +         cex.cor <- 0.4 / strwidth(txt)
    +     }
  +     text(0.5, 0.5, txt,
             +          cex = 1 + cex.cor * Cor) # Resize the text by level of correlation
  + }

pairs(df,upper.panel = panel.cor,lower.panel = panel.smooth)  #graphical view of correlation

#----- we see a strong correlation betwen bwt and gestion 
#--- lets run a SLR to see if the are linerly correlated

lm1 <- lm(df$bwt ~ df$gestation)
summary(lm1)

#--------------------- t-test---------------------
#Q1. is there a linear relation between birth weight and gestation? perform a t-test at
# alpha=0.05 level, construct and interpret 95% confidence interval.

#Determine t-value with df=998 and associated right hand tail probability of alpha/2=0.025
qt(0.025,998) = 1.962

#finding t value from summary to commary with above val
summary(lm1)

#now let calculate confidence interval
confint(lm1)

#-- lets plot the graph for bwt vs gestation to see the type of asscoiation
plot(df$gestation ,df$bwt)
abline(lm1,lty=1,col="blue")

#---------------------Global F-test-------------------------
#Q2. is there an linear relation between weight and height of mother on babies birth weight
# perform Global f-test with aplha = 0.05


#test stats for f-test
qf(0.95,df1=2,df2=997)

#find f-test from summary table
mlr2 <- lm(df$bwt ~ df$age + df$height)
summary(mlr2)


# now we have to perform testing on each indivdual parameter to check relative contribution 

#Q3. is height a significant predictor of birthweight after controlling age at alpha =0.05?


#Determine t-value with df=997 and associated right hand tail probability of alpha/2=0.025
qt(0.025,997) = 1.962

#find the t-value from summary table
lm2 <-lm(df$bwt ~ df$height)
summary(lm2)
#t = 5.898


#Q4. is age a significant predictor of birthweight after controlling height at alpha =0.05?


#Determine t-value with df=997 and associated right hand tail probability of alpha/2=0.025
qt(0.025,997) = 1.962

#find t-value from summary table
lm3 <-lm(df$bwt ~ df$age)
summary(lm3)
#t = 1.135

#--- lets do assumption test for lm1

#check for constant variance
plot(fitted(lm1),resid(lm1),axes=TRUE,frame.plot = TRUE,xlab="Fitted values",ylab = "Residuals")

#check for linearity
plot(df$bwt,resid(lm1),axes=TRUE,frame.plot = TRUE,xlab="Fitted values",ylab = "Residuals")

#check for normal distribution
hist(resid(lm1))

#Q5. Does birth weight depend on smoking habits?
glm3 <- glm(df$smoke ~ df$bwt,family = binomial(link = "logit"))
glm3

summary(glm3)

# Q6 Does any other parameter get impacted due to smoking habits?

glm4 <- glm(df$smoke ~ df$bwt + df$gestation + df$weight + df$height,family = binomial(link = "logit"))
glm4

summary(glm4)
