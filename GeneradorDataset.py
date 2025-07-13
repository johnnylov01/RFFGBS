import numpy as np
import pandas as pd
def generar_dataset(size_dataset=500, archivo="Dataset.csv",seed=42):
    edad=np.random.randint(2, 65, size=size_dataset)
    sexo=np.random.randint(0, 2, size=size_dataset)
        #0 normal, 1 heterocigoto, 2 homocigoto
    etnia=np.random.choice([0,1,2], size=size_dataset, p=[0.4, 0.3, 0.3])
    snps=[
        "rs1801274", "rs1800629", "rs1800630", "rs4986790",
        "HLA_DRB1_0401", "HLA_DRB1_1301", "HLA_DRB1_14",
        "rs7528684", "rs3761959", "rs2282284", "rs1800871", "rs1800872",
        "rs2397084", "rs1799969", "rs11003125", "rs2075820",
        "rs2066844", "rs2066845", "rs5743708", "rs121917864",
        "HLA_B54", "HLA_Cw1", "HLA_DRB4_0101", "KM1_KM3"]
    #Genotipos para SNPs
    genotype={
        snp: np.random.choice([0, 1, 2], size=size_dataset, p=[0.3, 0.4, 0.3]) for snp in snps
    }
    #DataFrame Base
    df=pd.DataFrame({
        "Edad": edad,
        "Sexo": sexo,
        "Etnia": etnia,
        **genotype
    })
    #Asignación de pesos mayores a los SNPs del metaanálisis
    snp_pesos={"rs1801274": 2.0, "rs1800629": 1.5, "rs1800630": 1.5, "rs4986790": 1.5,
               "HLA_DRB1_0401": 2.0, "HLA_DRB1_1301": 2.0, "HLA_DRB1_14": 2.0}
    #Riesgo ponderado
    df["riesgo"]=sum(df[snp]*snp_pesos.get(snp, 1.0) for snp in snps)
    #El riesgo aumenta 20% cada 10 años
    factorEdad=np.maximum(0, (df["Edad"] - 20) // 10 )
    df["riesgoEdad"]=factorEdad * 0.2
    df["RiesgoTotal"]=df["riesgo"] * (1+df["riesgoEdad"])
    #Umbral de riesgo
    umbral=df["RiesgoTotal"].quantile(0.75)
    df["GBS"]=(df["RiesgoTotal"] > umbral).astype(int)
    df.drop(columns=["riesgo", "riesgoEdad", "RiesgoTotal"], inplace=True)
    df.to_csv(archivo, index=False)
    print(f"Dataset generado EXITOSAMENTE en {archivo}")
    return df
if __name__ == "__main__":
    generar_dataset(size_dataset=1000, archivo="dataset_sintetico_GBS.csv")