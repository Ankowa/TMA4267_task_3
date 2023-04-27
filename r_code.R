# install.packages("FrF2", dependencies=TRUE)
library(FrF2)
library("tiff")
library("jpeg")

data <- read.csv("results.csv")
y <- data$Y
plan <- FrF2(nruns=16, nfactors=4, randomize=FALSE)
plan <- add.response(plan, y)
plan$A <- data$A; plan$B <- data$B; plan$C <- data$C; plan$D <- data$D

lm4 <- lm(y~.^4, data=plan)
summary(lm4)

width <- 6
height <- 6
res <- 500
tiff("main_effects_plot.tiff", units="in", width=width, height=height, res=res)
MEPlot(lm4)
dev.off()
img <- readTIFF("main_effects_plot.tiff", native=TRUE)
writeJPEG(img, target = "main_effects_plot.jpeg", quality = 1)
tiff("interaction_effects_plot.tiff", units="in", width=width, height=height, res=res)
IAPlot(lm4)
dev.off()
img <- readTIFF("interaction_effects_plot.tiff", native=TRUE)
writeJPEG(img, target = "interaction_effects_plot.jpeg", quality = 1)
tiff("residual_plot.tiff", units="in", width=width, height=height, res=res)
plot(lm4, which=1)
dev.off()
img <- readTIFF("residual_plot.tiff", native=TRUE)
writeJPEG(img, target = "residual_plot.jpeg", quality = 1)