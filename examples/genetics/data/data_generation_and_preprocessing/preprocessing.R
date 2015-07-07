library(kinship2)
    
ped <- read.table('./SeqSIMLA_results/data1.ped', header=FALSE)
haplotypes <- as.matrix(ped[,-c(1:6)])
# n is the number of rows, m is the number of columns
n <- nrow(haplotypes)
m <- ncol(haplotypes) / 2
# sum each two haplotype columns to get one genotype column
A <- haplotypes[, 2*(1:m)] + haplotypes[, 2*(1:m)-1] - 2 
# load the response vector
phe <- read.table('./SeqSIMLA_results/data1.phe', header=FALSE)
y <- phe[,3]

# get the kinship matrix
pedigreeinfo <- ped[ , 1:6]
fam <- pedigree(pedigreeinfo[,2], pedigreeinfo[,3], pedigreeinfo[,4], 
                pedigreeinfo[,5], pedigreeinfo[,6]-1, famid=pedigreeinfo[,1])
K <- kinship(fam)

# plot the pedigree of a family
jpeg("pedigree.jpeg")
plot(fam[2])
dev.off()

# save the matrices
write.table(A, file="design_matrix.csv", sep=",", row.names=F, col.names=F)
write.table(y, file="phenotype.csv", sep",", row.names=F, col.names=F)
write.table(as.matrix(K), file="kinship_matrix.csv", sep=",", row.names=F, col.names=F)
