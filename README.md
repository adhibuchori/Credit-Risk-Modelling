# Credit-Risk-Modelling
Final Task of ID/X Partners Data Scientist Project Based Internship on March 2024

Disusun Oleh : [Mochammad Adhi Buchori](www.linkedin.com/in/mochammad-adhi-b-2049a1136)

## 1. Domain Proyek

Domain proyek yang diambil untuk proyek *machine learning* ini, yaitu **Keuangan** dengan judul **Credit Risk Modeling**.

### 1.1. Latar Belakang

Credit risk atau risiko kredit merupakan salah satu bentuk risiko pasar yang paling banyak dianalisis dan sulit untuk diukur [1]. Risiko ini mengacu pada kemungkinan gagalnya peminjam untuk melunasi pinjaman atau utang mereka kepada pemberi pinjaman atau kreditur. Dalam konteks keuangan, risiko kredit menjadi sangat penting karena dapat mempengaruhi kesehatan finansial suatu institusi atau investor. Hal tersebut disebabkan karena kegagalan peminjam dalam menyelesaikan kewajibannya terkait pembayaran pinjaman dapat menimbulkan konsekuensi finansial yang substansial bagi pemberi pinjaman atau kreditur. Oleh karena itu, pengelolaan risiko kredit menjadi fokus utama dalam industri keuangan untuk memastikan keberlanjutan dan kestabilan operasional.

<div align="center">
  <img src="https://drive.google.com/uc?id=1X37Yt1FYUpteG2EzPUDotMeJmP63wL6y" alt="Credit Risk Modeling">
  <p>Gambar 1.1. Ilustrasi Credit Risk.</p>
</div>

Credit risk melibatkan penilaian risiko kredit dari pihak yang meminjam, termasuk penilaian kredit dan kemampuan perusahaan untuk membayar kembali pinjaman. Faktor-faktor yang mempengaruhi credit risk meliputi keadaan keuangan peminjam, kualitas manajemen, kondisi industri, kondisi ekonomi secara keseluruhan, serta berbagai faktor lain yang dapat mempengaruhi kemampuan peminjam untuk memenuhi kewajiban.

Manajemen credit risk menjadi fokus penting bagi lembaga keuangan dan investor untuk mengelola risiko yang terkait dengan portofolio kredit. Hal ini melibatkan penggunaan berbagai teknik, meliputi analisis kredit yang cermat, diversifikasi portofolio, penetapan batasan kredit, penggunaan instrumen derivatif untuk lindung nilai, dan pemantauan terus-menerus terhadap kualitas kredit secara berkala.

Pembangunan model yang dapat memprediksi credit risk sangat penting dalam konteks industri finansial, terutama untuk perusahaan peminjaman. Dengan adanya model ini, perusahaan dapat melakukan evaluasi yang lebih baik terhadap aplikasi pinjaman yang masuk, sehingga dapat mengurangi risiko default dan meningkatkan kepercayaan dari pihak investor. Model ini juga dapat membantu perusahaan dalam pengambilan keputusan yang lebih tepat dan efisien terkait dengan penentuan bunga, limit kredit, dan kebijakan peminjaman lainnya. Dengan menggunakan data pinjaman yang diterima dan ditolak, model dapat dikembangkan untuk mengidentifikasi pola-pola kredit yang mengarah pada risiko tinggi, sehingga perusahaan dapat mengambil langkah-langkah pencegahan yang diperlukan. Dengan adanya solusi ini, diharapkan perusahaan dapat meningkatkan kinerja operasionalnya dan memberikan layanan yang lebih baik kepada pelanggan serta pemangku kepentingan lainnya.

## 2. Business Understanding

Credit risk adalah probabilitas seorang peminjam akan mengalami kegagalan dalam membayar kembali jumlah pinjaman yang telah diberikan [2]. Dalam implementasinya, pemberian pinjaman dilakukan berdasarkan analisis kemampuan bisnis atau individu untuk memenuhi kewajiban pembayaran di masa mendatang, termasuk pembayaran pokok dan bunga. Dalam implementasinya, pemberi pinjaman melakukan langkah-langkah tertentu untuk memahami kondisi keuangan peminjam dan mengukur risiko bahwa peminjam tidak dapat memenuhi kewajiban pembayaran di masa yang akan datang.

Manajemen credit risk terdiri dari berbagai proses yang melibatkan beberapa langkah. Pada umumnya, proses ini dapat dikategorikan ke dalam 2 (dua) tahap utama, yaitu measurement (pengukuran) dan mitigation (mitigasi) [3]. Dalam hal ini, pengukuran melibatkan evaluasi keuangan dan profil peminjam untuk menilai risiko kredit, sedangkan mitigasi melibatkan penstrukturan pinjaman dan pengendalian portofolio untuk mengurangi risiko kredit. Berikut adalah diagram alur proses bisnis umum untuk mengajukan pinjaman:

<div align="center">
  <img src="https://drive.google.com/uc?id=1-Xk1r159TlSDBR2qqAiLB5OS_fgRmQm0" alt="Diagram Alur Proses Pengajuan Pinjaman">
  <p>Gambar 2.1.  Diagram Alur Proses Pengajuan Pinjaman.</p>
  <p><i>Source : Adapted from</i> [4]</p>
</div>

Berdasarkan gambar di atas, dapat diketahui bahwa secara umum proses pengajuan pinjaman terdiri dari tahapan berikut:

<div>
  <p align="center">Tabel 2.1. Proses Pengajuan Pinjaman.</p>
</div>

| Tahapan | Deskripsi |
|---|---|
| Permohonan Pinjaman | Klien mengajukan permohonan pinjaman dengan mengisi dokumen aplikasi pinjaman. |
| Pengecekan Klien | Sistem memeriksa apakah klien sudah terdaftar di sistem. |
| Dokumen Legal | Klien menyerahkan dokumen legal yang diperlukan. |
| Penilaian Aplikasi | Sistem melakukan penilaian terhadap aplikasi pinjaman berdasarkan dokumen yang diserahkan dan profil klien. |
| Negosiasi | Jika diperlukan, dilakukan negosiasi antara klien dan pihak peminjam terkait persyaratan pinjaman. |
| Persetujuan | Jika aplikasi disetujui, klien akan menerima persetujuan pinjaman dan perjanjian pinjaman. |
| Penandatanganan | Klien menandatangani perjanjian pinjaman. |
| Penyesuaian | Jika diperlukan, dilakukan penyesuaian terhadap jumlah pinjaman dan jadwal pembayaran. |
| Pencairan | Pinjaman dicairkan kepada klien. |
| Pembayaran | Klien melakukan pembayaran pinjaman sesuai dengan jadwal yang telah disepakati. |
| Pelunasan | Pinjaman dianggap lunas setelah seluruh pembayaran diterima. |
| Penolakan | Jika aplikasi ditolak, klien akan menerima pemberitahuan penolakan. |

### 2.1. Problem Statements

Berdasarkan latar belakang dan pemahaman bisnis yang telah diuraikan, proyek ini akan berfokus pada penyelesaian beberapa masalah, meliputi:
1. Bagaimana pendekatan dalam membersihkan data dan preprocessing untuk pemodelan credit risk?
2. Apa variabel penting yang akan dipertimbangkan dalam dataset credit risk dan bagaimana cara menangani data yang hilang?
3. Bagaimana cara menangani ketidakseimbangan kelas dalam dataset saat memilih model credit risk?
4. Bagaimana cara membandingkan pro dan kontra dari model machine learning yang akan digunakan untuk pemodelan credit risk?
5. Apa model machine learning yang paling efektif untuk memprediksi credit risk?

### 2.2. Goals

Tujuan dari proyek ini adalah sebagai berikut:
1. Membangun model machine learning yang dapat digunakan untuk memprediksi credit risk.
2. Membandingkan beberapa algoritma guna memperoleh akurasi terbaik dalam melakukan prediksi terhadap credit risk.
3. Mengidentifikasi variabel yang paling efektif dalam menentukan credit risk.

### 2.3. Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, peneliti mengembangkan model prediktif menggunakan 7 (tujuh) algoritma yang berbeda. Setiap model akan dievaluasi secara komprehensif untuk menentukan model yang paling optimal dalam memprediksi risiko kredit. Berikut adalah algoritma yang akan digunakan dalam pembangunan model prediksi:

1. Random Forest  
Random Forest adalah sebuah algoritma machine learning yang bekerja dengan menggabungkan output dari beberapa decision tree untuk menghasilkan 1 (satu) model prediktif yang optimal [5]. Random Forest terkenal dengan kemudahan penggunaan dan fleksibilitasnya, sehingga banyak diminati.  Algoritma ini dapat menangani permasalahan klasifikasi maupun regresi.
2. Gradient Boosting    
Gradient Boosting adalah metode ensemble machine learning yang digunakan untuk membangun model prediksi secara berurutan dengan menggabungkan ensemble decision tree yang relatif lemah menjadi model yang lebih kuat [6]. Teknik ini bertujuan untuk meningkatkan kinerja prediksi secara keseluruhan dengan mengoptimalkan bobot model berdasarkan kesalahan iterasi sebelumnya. Dengan demikian, kesalahan prediksi berkurang secara bertahap dan akurasi model pun meningkat.
3. AdaBoost  
AdaBoost adalah algoritma ensemble learning yang digunakan untuk meningkatkan kinerja model prediksi dengan menggabungkan beberapa model lemah menjadi model yang lebih kuat [7].
4. XGBoost  
XGBoost adalah toolkit distributed gradient boosting yang telah disesuaikan untuk pelatihan yang efisien dan scalable dari model machine learning [8]. Algoritma ini menggunakan decision trees sebagai base learners dan menerapkan teknik regularisasi untuk meningkatkan generalisasi model [9]. Dikenal karena efisiensi komputasinya, analisis pentingnya variabel, dan penanganan nilai-nilai yang hilang, XGBoost banyak digunakan untuk tugas-tugas seperti regresi, klasifikasi, dan ranking.
5. Logistic Regression  
Logistic Regression adalah algoritma regresi yang digunakan untuk memodelkan probabilitas bahwa suatu data masuk ke dalam kelas tertentu, khususnya ketika variabel dependen bersifat dichotomous (biner) [10]. Meskipun disebut "regresi," Logistic Regression sebenarnya digunakan untuk tugas klasifikasi biner, di mana outputnya adalah probabilitas data termasuk ke dalam kelas target. Algoritma ini menggunakan fungsi sigmoid untuk menghasilkan output yang berada di antara 0 dan 1, yang dapat diinterpretasikan sebagai probabilitas kelas tertentu.
6. K-Nearest Neighbors  
K-Nearest Neighbors adalah algoritma klasifikasi (dan juga regresi) yang bekerja berdasarkan prinsip "majority voting" untuk klasifikasi atau rata-rata dari tetangga terdekat [11]. Algoritma ini menghitung jarak antara data yang ingin diprediksi dengan data training dan memilih kelas mayoritas (atau nilai rata-rata) dari tetangga terdekat (dengan jumlah tetangga yang ditentukan oleh nilai K). KNN merupakan algoritma non-parametrik yang cukup sederhana, namun dapat memberikan hasil yang baik terutama dalam kasus dataset yang relatif kecil dan dimensi yang rendah.
7. Neural Network  
Neural Network (jaringan saraf tiruan) adalah model komputasi yang terinspirasi dari struktur dan fungsi jaringan saraf biologis dalam otak manusia [12]. Tujuan utama Neural Network adalah untuk memproses informasi dan melakukan tugas-tugas seperti klasifikasi, regresi, pengenalan pola, dan lainnya, dengan cara yang mirip dengan cara otak manusia memproses informasi.

## 3. Data Loading

Tahap ini meliputi proses mengimpor atau memuat data untuk analisis atau pemrosesan lebih lanjut. Tujuannya adalah untuk membuat data yang diperlukan tersedia dalam format yang sesuai untuk analisis atau pengolahan lebih lanjut.

## 4. Data Understanding

<div align="center">
  <img src="https://drive.google.com/uc?id=1nBLWhn90Se7NO935kR93Ql3fIe6ilQTk" alt="Loan Dataset">
  <p>Gambar 4.1. Ilustrasi Loan Dataset.</p>
</div>

Dataset yang digunakan dalam proyek ini, yaitu data pinjaman yang disediakan oleh Rakamin Academy sebagai bagian dari program Project Based Internship. Data tersebut terdiri dari 466.285 entri dengan 75 kolom. Untuk mengakses dataset yang digunakan dalam proyek ini, silakan kunjungi tautan berikut (*Source* : [*Link Dataset*](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv)).

### 4.1. Data Dictionary

Pada tahap ini, kamus data dibuat untuk memahami konteks setiap variabel dalam dataset pinjaman (*loan dataset*). Selain itu, juga dilakukan identifikasi untuk menghapus fitur yang tidak relevan dengan kebutuhan pemodelan. Langkah-langkah ini bertujuan untuk meningkatkan efektivitas dan profesionalisme dalam proses analisis data.

Berikut merupakan kamus data yang disusun untuk membantu memahami konteks setiap variabel dalam *loan dataset*:

| Variabel | Deskripsi |
|---|---|
| **id** | ID unik yang ditetapkan LC untuk listing pinjaman. |
| **member_id** | ID unik yang ditetapkan LC untuk anggota peminjam. |
| loan_amnt | Jumlah pinjaman yang diajukan oleh peminjam. Jika departemen kredit mengurangi jumlah pinjaman, nilainya akan tercermin di sini. |
| funded_amnt | Jumlah total yang dikomitkan untuk pinjaman tersebut pada saat itu. |
| funded_amnt_inv | Jumlah total yang dikomitkan oleh investor untuk pinjaman tersebut pada saat itu. |
| term | Jumlah pembayaran pinjaman. Nilai dalam bulan dan bisa 36 atau 60. |
| int_rate | Suku Bunga pinjaman |
| installment | Pembayaran bulanan yang terutang oleh peminjam jika pinjaman berasal. |
| grade | Tingkat pinjaman yang ditetapkan LC |
| sub_grade | Subtingkat pinjaman yang ditetapkan LC |
| emp_title | Jabatan yang diberikan oleh Peminjam saat mengajukan pinjaman. |
| emp_length | Lamanya bekerja dalam tahun. Nilai kemungkinan antara 0 dan 10 dimana 0 berarti kurang dari satu tahun dan 10 berarti sepuluh tahun atau lebih. |
| home_ownership | Status kepemilikan rumah yang diberikan oleh peminjam saat pendaftaran. Nilai yang mungkin: SEWA, MILIK, KPR, LAINNYA. |
| annual_inc | Penghasilan tahunan yang dilaporkan sendiri oleh peminjam saat pendaftaran. |
| verification_status | Menunjukkan apakah pendapatan diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi. |
| **issue_d** | Bulan dimana pinjaman didanai |
| loan_status | Status pinjaman saat ini |
| pymnt_plan | Menunjukkan apakah rencana pembayaran telah ditetapkan untuk pinjaman |
| **url** | URL untuk halaman LC dengan data listing. |
| desc | Deskripsi pinjaman yang diberikan oleh peminjam |
| purpose | Kategori yang diberikan oleh peminjam untuk permintaan pinjaman. |
| title | Judul pinjaman yang diberikan oleh peminjam |
| zip_code | 3 digit awal dari kode pos yang diberikan oleh peminjam dalam aplikasi pinjaman. |
| addr_state | Negara bagian yang diberikan oleh peminjam dalam aplikasi pinjaman |
| dti | Rasio yang dihitung menggunakan total pembayaran hutang bulanan peminjam pada total kewajiban hutang, tidak termasuk hipotek dan pinjaman LC yang diminta, dibagi dengan pendapatan bulanan yang dilaporkan sendiri oleh peminjam. |
| **delinq_2yrs** | Jumlah kejadian terlambat pembayaran 30+ hari dalam catatan kredit peminjam selama 2 tahun terakhir. |
| earliest_cr_line | Bulan dimana jalur kredit terawal peminjam dibuka |
| **inq_last_6mths** | Jumlah penyelidikan dalam 6 bulan terakhir (tidak termasuk penyelidikan auto dan hipotek) |
| **mths_since_last_delinq** | Jumlah bulan sejak keterlambatan pembayaran terakhir peminjam. |
| **mths_since_last_record** | Jumlah bulan sejak catatan publik terakhir. |
| open_acc | Jumlah jalur kredit terbuka dalam catatan kredit peminjam. |
| pub_rec | Jumlah catatan publik yang merugikan |
| revol_bal | Total saldo kredit bergulir |
| revol_util | Rasio penggunaan kredit bergulir, atau jumlah kredit yang digunakan peminjam relatif terhadap semua kredit bergulir yang tersedia. |
| total_acc | Jumlah total jalur kredit yang saat ini ada di file kredit peminjam |
| initial_list_status | Status pencatatan awal untuk pinjaman. Kemungkinan nilainya adalah - Keseluruhan, Sebagian |
| **out_prncp** | Sisa pokok terutang untuk total keseluruhan yang didanai |
| **out_prncp_inv** | Sisa pokok terutang untuk porsi dari total keseluruhan yang didanai oleh investor |
| **total_pymnt** | Pembayaran yang diterima hingga saat ini untuk total keseluruhan yang didanai |
| **total_pymnt_inv** | Pembayaran yang diterima hingga saat ini untuk porsi dari total keseluruhan yang didanai oleh investor |
| **total_rec_prncp** | Pokok yang diterima hingga saat ini |
| **total_rec_int** | Bunga yang diterima hingga saat ini |
| **total_rec_late_fee** | Biaya keterlambatan yang diterima hingga saat ini |
| **recoveries** | Pemulihan bruto setelah penghapusan buku |
| **collection_recovery_fee** | Biaya pemulihan setelah penghapusan buku |
| **last_pymnt_d** | Bulan terakhir pembayaran diterima |
| **last_pymnt_amnt** | Jumlah total pembayaran terakhir yang diterima |
| **next_pymnt_d** | Tanggal pembayaran terjadwal berikutnya |
| **last_credit_pull_d** | Bulan terakhir LC menarik kredit untuk pinjaman ini |
| **collections_12_mths_ex_med** | Jumlah penagihan dalam 12 bulan tidak termasuk penagihan medis |
| **mths_since_last_major_derog** | Bulan sejak peringkat terburuk 90 hari atau lebih terakhir |
| policy_code | Kode kebijakan yang tersedia untuk umum: policy_code=1 produk baru yang tidak tersedia untuk umum policy_code=2 |
| application_type | Menunjukkan apakah pinjaman tersebut merupakan pengajuan individu atau pengajuan bersama dengan dua peminjam bersama |
| acc_now_delinq | Jumlah akun di mana peminjam saat ini terlambat. |
| **tot_coll_amt** | Total jumlah penagihan yang pernah terutang |
| **tot_cur_bal** | Total saldo saat ini dari semua akun |
| **total_rev_hi_lim** | Total batas kredit/kredit limit tertinggi untuk kredit bergulir |

## 5. Exploratory Data Analysis

Exploratory Data Analysis (EDA) adalah proses investigasi awal yang dilakukan pada dataset untuk memahami dan menganalisis karakteristik utama dalam dataset. Tujuan dari EDA adalah untuk mengidentifikasi pola, hubungan, anomali, dan informasi penting lainnya dalam dataset tanpa membuat asumsi atau hipotesis terlebih dahulu. Metode yang umum digunakan dalam EDA meliputi visualisasi data, statistik deskriptif, dan teknik analisis lainnya untuk mendapatkan pemahaman yang mendalam tentang data sebelum melakukan analisis lebih lanjut atau membangun model prediktif.

### 5.1. Deskripsi Variabel

Tahap ini merujuk pada proses analisis yang bertujuan untuk memahami struktur, karakteristik, dan informasi yang terkandung dalam variabel-variabel yang digunakan dalam suatu dataset.

### 5.2. Feature Engineering

Pada tahap ini, fitur-fitur baru diciptakan dari data yang sudah ada untuk meningkatkan performa model machine learning.

### 5.3. Data Analysis

#### 5.3.1. Univariate Analysis

Univariate Analysis adalah sebuah metode analisis statistik yang digunakan untuk memahami karakteristik dari satu variabel tunggal dalam suatu dataset. Tujuan utama dari analisis univariat adalah untuk merangkum dan menyajikan data, serta mendapatkan wawasan yang lebih dalam tentang distribusi, pola, dan sifat-sifat statistik dari variabel.

#### 5.3.2. Multivariate Analysis

Multivariate Analysis adalah sebuah pendekatan statistik yang digunakan untuk memahami hubungan antara dua atau lebih variabel dalam sebuah dataset. Berbeda dengan analisis univariat yang hanya fokus pada satu variabel tunggal, analisis multivariat memungkinkan pengguna untuk mengeksplorasi korelasi, pola, dan struktur yang kompleks antara beberapa variabel.

## 6. Data Preparation

Data Preparation adalah proses persiapan data sebelum data dapat digunakan untuk analisis atau pemodelan. Tujuannya adalah untuk memastikan data siap digunakan dalam analisis atau pemodelan. Berikut merupakan teknik yang peneliti gunakan pada tahap Data Preparation:
1. Handle Missing Data
2. Convert Numeric Variable
3. Remove Outliers
4. Encode Data
5. Train Test Split

### 6.1. Handle Missing Data

Pada tahap ini, dilakukan identifikasi, analisis, dan pembersihan nilai kosong dalam dataset untuk memastikan konsistensi dan keandalan data yang digunakan dalam analisis atau pemodelan. Langkah-langkah ini bertujuan untuk memastikan kualitas dan akurasi data yang digunakan dalam proses selanjutnya.

### 6.2. Convert Numeric Variables

Pada tahap ini, dilakukan pemeriksaan dan penyesuaian tipe data kolom-kolom dalam DataFrame df untuk memastikan data dapat diproses sesuai dengan kebutuhan analisis atau pemodelan yang akan dilakukan. Langkah ini penting untuk memastikan konsistensi dan akurasi data yang digunakan.

### 6.3. Remove Outliers

Pada tahap ini, dilakukan deteksi outlier dalam dataset numerik, diikuti dengan penggantian nilai outlier dengan NaN, dan penghapusan baris yang mengandung nilai NaN. Tindakan ini bertujuan untuk membersihkan data dari outlier sehingga data yang digunakan untuk analisis atau pemodelan yang lebih konsisten.

### 6.4. Encode Data

Pada tahap ini, dilakukan konversi variabel kategorikal dalam dataset menjadi representasi numerik yang sesuai agar dapat dimengerti dan diproses oleh algoritma machine learning. Tahap ini melibatkan penggunaan teknik One-Hot-Encoding guna menyesuaikan data kategorikal menjadi format yang dapat dipahami oleh model machine learning.

#### 6.4.1. Define Target

Pada tahap ini, target variabel (`loan_status`) dalam dataset didefinisikan, dipersiapkan, dan dikonversi dari representasi teks menjadi representasi numerik menggunakan LabelEncoder.

#### 6.4.2. Convert Categorical Variables

Pada tahap ini, variabel kategorikal dalam DataFrame diubah menjadi representasi numerik yang dapat digunakan untuk analisis lebih lanjut, terutama dalam konteks pemodelan data.

#### 6.4.3. Create Transformer

Pada tahap ini, dilakukan preprocessing data pada DataFrame df dan hasil preprocessing disatukan dengan variabel target `loan_status` sehingga data siap untuk analisis lanjutan atau pembuatan model machine learning. Proses tersebut melibatkan penearapan transformasi pada kolom-kolom dalam DataFrame. Transformasi yang ditentukan adalah one-hot encoding untuk kolom-kolom kategorikal dan penskalaan min-max untuk kolom-kolom numerik.

### 6.5. Train Test Split

Pada tahap ini, data dibagi menjadi subset latih dan uji, distribusi kelas dalam variabel target diperiksa, dan fitur serta target disiapkan untuk pelatihan dan pengujian model machine learning.

## 7. Modelling

Pada tahap ini, proses pembangunan dan penyesuaian model dilakukan berdasarkan data yang tersedia untuk tujuan analisis, prediksi, atau pengambilan keputusan. Model-model ini digunakan untuk mengidentifikasi pola, hubungan, atau tren dalam data, serta untuk membuat prediksi atau estimasi berdasarkan informasi yang ada. Langkah-langkah dalam proses modelling meliputi pemilihan model yang sesuai, pelatihan model menggunakan data latih, evaluasi kinerja model menggunakan data uji, dan penyesuaian model untuk meningkatkan kinerja dan akurasi.

Berikut merupakan hasil visualisasi perbandingan skor akurasi untuk setiap model:

<div align="center">
  <img src="https://drive.google.com/uc?id=1EaWVOTkirhGWrJ2djMmdja9NWsXlhiZY" alt="Accuracy Score">
  <p>Gambar 7.1. Hasil Visuailsasi Perbandingan Skor Akurasi untuk Setiap Model.</p>
</div>

Dengan demikian, dapat disimpulkan bahwa mayoritas algoritma berhasil mencapai tingkat akurasi sebesar 81%, dengan pencapaian tertinggi terdapat pada algoritma AdaBoost yang mencapai 81.59%. Hal ini menunjukkan bahwa mayoritas dari model-model tersebut mampu mengklasifikasikan dengan benar sekitar 81% kasus pinjaman.

## 8. Evaluation

Pada tahap ini, dilakukan proses pengukuran kinerja dan akurasi model yang telah dibangun berdasarkan data yang digunakan untuk pelatihan. Tujuan dari evaluasi adalah untuk mengevaluasi seberapa baik model tersebut dapat memprediksi atau mengeneralisasi pola dari data baru yang belum pernah dilihat sebelumnya. Dalam hal ini, evaluasi akan dilkaukan dengan menggunakan Confusion Matrix dan ROC Curve and AUC. Implementasi confusion matrix dilakukan dengan melibatkan penggunaan matriks yang tercantum dalam classification report, seperti akurasi, presisi, recall, dan F-1 Score. Tentunya, evaluasi yang baik membantu memastikan bahwa model dapat memberikan hasil yang optimal.

### 8.1. Confusion Matrix

Berikut merupakan hasil visualisasi Confusion Matrix untuk setiap algoritma:

<div align="center">
  <img src="https://drive.google.com/uc?id=1izg_ARpdFkXehMiMSUmzw4cq2RCj7q2g" alt="Accuracy Score">
  <p>Gambar 8.1. Hasil Visuailsasi Confusion Matrix untuk Setiap Model.</p>
</div>

#### 8.1.1. Classification Report

Berikut merupakan laporan klasifikasi untuk setiap algoritma:

<div align="center">
  <img src="https://drive.google.com/uc?id=1BIxhpgw8AA6xjdQgSzFdGWdEUcSqKxmk" alt="Accuracy Score">
  <p>Gambar 8.1.1. Laporan Klasifikasi untuk Setiap Algoritma.</p>
</div>

Berdasarkan hasil visualisasi Confusion Matrix dan Laporan Klasifikasi, dapat disimpulkan bahwa:

1. Terdapat ketidakseimbangan kelas di mana terdapat lebih banyak kasus `Fully Paid` (kelas 1) daripada kasus `Charged Off` (kelas 0).
2. Seluruh model menunjukkan recall yang sangat tinggi (1.00) untuk kelas `Fully Paid` (kelas 1), yang menujukkan bahwa seluruh model hampir secara sempurna mengidentifikasi pinjaman yang baik.
3. Presisi untuk kelas `Charged Off` rendah (sekitar 0.55) untuk semua model. Ini menunjukkan bahwa model mungkin salah mengklasifikasikan banyak pinjaman yang buruk (charged off) sebagai pinjaman yang baik (fully paid).

### 8.2. ROC Curve and AUC

Berikut merupakan hasil visualisasi kurva ROC (Receiver Operating Characteristic) dan nilai AUC (Area Under the ROC Curve) untuk setiap algoritma:

<div align="center">
  <img src="https://drive.google.com/uc?id=1CbzOot7hldawmAtOfkwBGziasCzpDrh0" alt="Accuracy Score">
  <p>Gambar 8.2. Hasil Visualisasi Kurva ROC (Receiver Operating Characteristic) dan Nilai AUC (Area Under the ROC Curve) untuk Setiap Algoritma.</p>
</div>

Berdasarkan hasil visualisasi kurva ROC (Receiver Operating Characteristic) dan nilai AUC (Area Under the ROC Curve), dapat disimpulkan bahwa:

1. Seluruh model kecuali KNeighborsClassifier memiliki skor AUC yang serupa sekitar 0.7. Ini menunjukkan kemampuan moderat untuk membedakan antara pinjaman yang baik dan buruk.
2. LogisticRegression memiliki AUC tertinggi (0.702475). Ini menunjukkan kinerja yang sedikit lebih baik dalam membedakan jenis pinjaman.
3. KNeighborsClassifier memiliki AUC yang jauh lebih rendah (0.559735). Ini menunjukkan kinerja yang buruk dalam membedakan jenis pinjaman.

## 9. Kesimpulan

### 9.1. Kesimpulan
Dengan demikian, dapat disimpulkan bahwa model risiko kredit cenderung memiliki kinerja yang lebih baik dalam mengidentifikasi pinjaman yang baik daripada yang buruk. Kecenderungan ini dipengaruhi oleh ketidakseimbangan antara kelas dalam data yang digunakan untuk pelatihan model. Selain itu, Skor AUC menunjukkan bahwa model memiliki kemampuan yang cukup baik dalam menilai risiko kredit. Dengan meningkatkan skor AUC dan mempertimbangkan metrik lain, lending company (LC) dapat mengembangkan model yang lebih akurat dan sesuai dengan kebutuhan bisnis.

### 9.2. Saran

1. **Data Balancing**  
Teknik pengelolaan ketidakseimbangan data dapat diterapkan untuk meratakan proporsi kasus positif dan negatif dalam dataset pelatihan.
2. **Tuning Model**  
Proses penyetelan model dapat dilakukan untuk menyesuaikan hiperparameter algoritma guna meningkatkan kinerja terutama pada kelas minoritas, seperti kasus `Charged Off`.
3. **Business Recommendation**  
Dapat dilakukan pembuatan business recommendation berdasarkan data yang tersedia untuk mengembangkan potensi bisnis.

## 10. Referensi

[1]	G. Carlone, Introduction to Credit Risk. CRC Press, 2020.

[2]	N. Arora and P. D. Kaur, “A Bolasso based consistent feature selection enabled random forest classification algorithm: An application to credit risk assessment,” Applied Soft Computing, vol. 86, p. 105936, Jan. 2020, doi: 10.1016/j.asoc.2019.105936.

[3]	K. Peterdy, “Credit Risk,” Corporate Finance Institute. Accessed: Mar. 24, 2024. [Online]. Available: https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/

[4]	G. Razis and S. Mitropoulos, “An integrated approach for the banking intranet/extranet information systems: the interoperability case,” International Journal of Business and Systems Research, vol. 1, no. 1, p. 1, 2022, doi: 10.1504/ijbsr.2022.10031295.

[5]	S. E. R, “Building a Random Forest Model: A Step-by-Step Guide,” Analytics Vidhya, Jun. 17, 2021. https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/ (accessed Mar. 24, 2024).

[6]	A. Saini, “Gradient Boosting: A Step-by-Step Guide,” Analytics Vidhya, Sep. 20, 2021. https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/ (accessed Mar. 24, 2024).

[7]	“AdaBoost Algorithm in Machine Learning,” AlmaBetter, Apr. 12, 2023. Accessed: Mar. 24, 2024. [Online]. Available: https://www.almabetter.com/bytes/tutorials/data-science/adaboost-algorithm.

[8]	“XGBoost Algorithm in Machine Learning,” AlmaBetter, Apr. 12, 2023. Accessed: Mar. 24, 2024. [Online]. Available: https://www.almabetter.com/bytes/tutorials/data-science/xgboost-algorithm    

[9]	guest_blog, “What Is XGBoost and How Does It Improve Machine Learning?,” Analytics Vidhya, Sep. 06, 2018. https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/ (accessed Mar. 24, 2024).

[10]	“What is Logistic Regression?,” Statistics Solutions, Dec. 21, 2010. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/ (accessed Mar. 25, 2024).

[11]	A. Christopher, “K-Nearest Neighbor - The Startup - Medium,” The Startup, Feb. 03, 2021. Accessed: Mar. 25, 2024. [Online]. Available: https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4    

[12]	J. Chen, “What Is a Neural Network?,” Investopedia, Sep. 12, 2006. Accessed: Mar. 25, 2024. [Online]. Available: https://www.investopedia.com/terms/n/neuralnetwork.asp
