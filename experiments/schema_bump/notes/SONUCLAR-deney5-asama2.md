# Deney 5 — Aşama 2 sonuçları (gerçek model)

Tarih: 22 Eylül 2026
Model: `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM, reasoning kapalı)
Korpus: 40 koşu × ~61 olay, `SEED=20260922`
Kod: `deney5_asama2.py`, `deney5_mem0.py`, `deney5_kollar.py`

> **Koşu bütünlüğü uyarısı.** A/C/D/E kolları ile B (Mem0) kolu **aynı koşudan
> gelmiyor**. Ölçüm düzeltmeleri esas olarak Mem0 satırını etkilediği için o kol
> ayrıca koşuldu; A/C/D/E iki bağımsız koşuda da aynı değerleri verdi. Aynı
> korpus, aynı tohum, aynı model kullanıldı. Yine de tek koşudan gelmiş gibi
> sunulmamalı.

---

## Tablo

| kol | validity@v2 | stored KATI | stored GEVŞEK | retrieved | n |
|---|---|---|---|---|---|
| D supersession | **100%** | 100% | 100% | 100% | 40 |
| A no-memory (tavan) | **95.0%** | 100% | 100% | 100% | 40 |
| **B mem0** | **20.0%** | 13.3% | **90.9%** | 26.7% | **15** |
| C tfidf+recency | 2.5% | 100% | 100% | **0%** | 40 |
| E lossy | 0% | 0% | 0% | 0% | 40 |

`validity@v2` = ajanın ürettiği çağrı v2 şemasına uyuyor mu. Ana skor.

**Mem0, hafızasız çalışmanın 75 puan altında.** Bu, MemoryAgentBench'teki
"Mem0 32.4 vs kendi backbone'u 82.4" bulgusunun bağımsız bir görevde tekrarı.

Mem0 satırı 15 koşu (5 parçalı koşu, 20 rpm, **sıfır yazma hatası**);
diğerleri 40 koşu.

---

## Bulgular

### 1. Tavan yüksek, yani kayıplar hafızadan geliyor

Ajan tüm geçmişi gördüğünde **%95** doğru çağrı üretiyor. Bu, deneyin geri
kalanını okunabilir kılan sayı: hafıza kollarındaki her düşüş modelin
yetersizliğinden değil, katmanın ona ne verdiğinden kaynaklanıyor.

Aşama 1'de deterministik ajanla bu değer %100'dü. Aradaki 5 puan gerçek
modelin kendi hata payı.

### 2. Deterministik politika tavanı geçiyor

"En son şema kazanır" diyen yirmi satırlık politika: **%100**.
Tam bağlamdan (%95) daha iyi — çünkü ajana çelişkisiz, tek bir doğru
tanım veriyor. Gürültü azaltmak, bilgi eklemekten değerli.

FactConsolidation'da aynı politika kapsam içi %96.6–100 vermişti.
İki bağımsız görevde aynı sonuç.

### 3. Benzerlik araması bu görevde çalışmıyor

TF-IDF + recency: bilgi deposunda **%100**, ajana ulaşma **%0**,
sonuç **%2.5**. Şema değişikliği tek bir olay; yüzlerce rutin çağrının
arasında benzerlik sıralamasında öne çıkamıyor. Recency de kurtarmıyor,
çünkü değişim "yakın geçmişte" değil orta geçmişte.

---

## Ölçüm düzeltmeleri — ilk tam koşu neden geçersizdi

İlk koşuda Mem0 satırı `stored=7.5%` ama `retrieved=25%` verdi.
**Saklanmayan getirilemez**; bu, ölçümün bozuk olduğunun kanıtıydı.
Dört ayrı sorun çıktı, dördü de bende:

| # | sorun | etki | düzeltme |
|---|---|---|---|
| 1 | `get_all()` varsayılan limitle depoyu kırpıyordu | stored olduğundan düşük | limitsiz okuma |
| 2 | 8 yazma çağrısı 503 ile düştü | servis hatası ürünün skoruna yazılıyordu | Mem0'ın iç LLM çağrılarına retry |
| 3 | `stored` ölçütü tam imza arıyordu | ürün başka biçimde saklayınca sıfırlıyordu | katı + gevşek, ikisi de raporlanıyor |
| 4 | slot kararları kaydedilmiyordu | "üzerine yazıldı" ile "hiç yazılmadı" ayrılamıyordu | ADD/UPDATE/DELETE loglanıyor |

(3) ve (4), X'te @kartikb753'ün sorusundan çıktı:
*"do you log which memory slot each write touched?"* — hayır, loglamıyordum,
ve teşhis edememe sebebim buydu.

---

## Mem0 kolunun koşu geçmişi

Dört deneme, dört farklı kesilme sebebi. Üçü bende, biri makinede:

| # | n | sonuç | sebep |
|---|---|---|---|
| 1 | 40 | geçersiz | ölçüm bozuk (`stored < retrieved`), dört düzeltme yapıldı |
| 2 | 40 | geçersiz | tüm yazmalar OpenRouter'a gitti, `402` — ortam değişkeni config'i ezdi |
| 3 | 15 | kesildi | 429: Mem0 çağrıları hız sınırından geçmiyordu, düzeltildi |
| 4 | 15 | kesildi | sistem belleği |
| 5 | 15 | kesildi | sistem belleği (bellek payı yeterli sanılmıştı — bkz. aşağı) |

Geçerli kalan ölçüm: **5 koşuluk pilot**, düzeltilmiş ölçütlerle, NVIDIA
üzerinden, OpenRouter karışmadan önce. Aşağıdaki bulgu oradan.

**Bellek tahmini neden tuttu sanıldı:** Mem0Arm'ın ayak izi ölçüldü —
kurulum 256 MB, 20 olay sonrası 397 MB. Buradan 15 koşu (915 olay) için
"0.5–0.7 GB yeter" sonucuna varıldı. Bu bir **ölçüm değil, ekstrapolasyondu**:
20 olaydan 915 olaya büyümenin doğrusal olmadığı varsayıldı ama doğrulanmadı.
Koşu başladığında 3.89 GB boştu, koşu sırasında 1.3 GB'a indi.
Ölçülen ile varsayılan arasındaki fark, bu deneyin başına üçüncü kez geliyor.

## ⚠ 30 Eylül düzeltmeleri — bu bölümü okumadan aşağıdakini okuma

Bir hafta sonra yapılan kontrolde üç şey ortaya çıktı. İkisi ölçüm hatası,
biri **bulgunun anlamını değiştiriyor**.

### 1. `limit` parametresi sessizce yok sayılıyordu (ölçüm hatası)

mem0 2.1.0'da `get_all` ve `search`'ün parametresi `top_k` (varsayılan 20),
`limit` değil. Verilen `limit=10000` ve `limit=12` `**kwargs`'a düşüp hata
vermeden yok sayıldı:

- `get_all` ~50 kaydın **yalnızca 20'sini** döndürdü → `stored KATI` ve
  `stored GEVŞEK` sütunları deponun ~%40'lık kesitinden ölçüldü.
  `stored < retrieved` çelişkisinin sebebi buydu: teşhiste v2 imzalı kayıt
  `search`'ün ilk 20'sindeydi, `get_all`'un ilk 20'sinde değildi.
- `search` 12 değil **20** kayıt döndürdü → Mem0 kolunda ajana verilen
  bağlam raporda yazılandan büyük.

Etkilenmeyenler: `validity@v2` (ajan gerçekten 20 kayıtla çalıştı, ölçülen
buydu) ve slot kararları (`add()` dönüşünden, `get_all`'dan değil).
Kodda `top_k` ile düzeltildi; `stored` sütunları bu raporda **güvenilmez**.

### 2. ADD-only, Mem0'ın belgelenmiş tasarım kararı (anlam değişikliği)

"754 yazma, sıfır UPDATE" doğru ölçüldü — ama bu, Mem0'ın her durumda
karar vermemesi değil, **v2.0.0 ile (2026-04-14) bilinçli olarak ADD-only'ye
geçmesi**:

> "Single-Pass Extraction: Replaced 2-LLM-call pipeline with additive
> extraction … Memories accumulate via linked_memory_ids: no more
> UPDATE/DELETE events (#4805)." — docs.mem0.ai/changelog/sdk

> "The ADD-only model means memories accumulate over time. When information
> changes, the new fact is stored alongside the old one. Retrieval handles
> ranking: the most relevant, current information surfaces first."
> — docs.mem0.ai/migration/oss-v2-to-v3

Kaynak kodda da doğrulandı: `add()` hattı (Phase 0–8) yalnızca `"ADD"`
üretiyor; `UPDATE`/`DELETE` sadece çağıranın elle kullandığı `update()`/
`delete()` metodlarında.

Yani doğru soru "Mem0 neden güncellemiyor" değil — geçersiz kılmayı yazmadan
**okumaya taşımış**. Doğru soru: **okuma gerçekten güncel bilgiyi öne çıkarıyor mu?**

### 3. Açık kaynak sıralamada zaman sinyali yok (kaynaktan doğrulandı)

`mem0/utils/scoring.py` → `score_and_rank(semantic_results, bm25_scores,
entity_boosts, threshold, top_k)`. Fonksiyonda `created_at`, `updated_at`,
zaman ya da yenilik terimi yok. Varlık bonusu aynı varlığa bağlı tüm
kayıtlara eşit uygulanıyor ve kayıt sayısıyla küçülüyor
(`1/(1+0.001·(n−1)²)`): eskiyi yeniden ayırmıyor.

Zaman odaklı özellikler yalnızca ücretli Platform'da:
- `reference_date`: *"Platform-only temporal parameter. Not supported in OSS."*
- `decay=True`: açık kaynakta hata fırlatıyor.

**Sonuç:** belgedeki "current information surfaces first" iddiası, açık kaynak
sürümde sıralamaya giren hiçbir mekanizmayla desteklenmiyor. Eski ve yeni
bilgi yalnızca metin benzerliğiyle yarışıyor. Platform sürümü test edilmedi.

### 4. MemoryAgentBench farklı bir Mem0 algoritmasını ölçtü

MAB (arXiv 2507.05257, Temmuz 2025) v2.0.0 öncesi, UPDATE/DELETE yapan
2-LLM-çağrılı hattı test etti. Deney 5 yeni ADD-only hattı test etti.
İki bulgu **farklı ürün sürümleri** hakkında; birbirine bağlanırken bu
belirtilmeli.

---

## Mem0: asıl bulgu — ADD-only hafıza neyi taşıyamaz

15 koşu, sıfır yazma hatası, 754 yazma kararı:

```
SLOT KARARLARI: {'ADD': 754}     UPDATE: 0     DELETE: 0
```

Mutasyon tipine göre ayrılınca mekanizma görünüyor:

| mutasyon | değişimin doğası | Mem0 validity |
|---|---|---|
| `type_change` | alan tipi değişiyor | **2/2 (100%)** |
| `add_required` | yeni zorunlu alan | 1/3 (33%) |
| `rename` | eski ad **geçersiz** oluyor | **0/5** |
| `remove` | alan **kaldırılıyor** | **0/4** |
| `split` | eski alan **bölünüyor** | **0/1** |

**Bilgi EKLEYEN değişimlerde Mem0 başarılı olabiliyor.
Bilgi GEÇERSİZ KILAN değişimlerde 0/10.**

Sebep yapısal ve 754 ADD ile birebir örtüşüyor:
**bir şeyin artık geçerli olmadığını "ekleyerek" anlatamazsın.**
`rename`'de yeni adı eklemek yetmez — eski adın kullanılmaması gerektiği
bilgisi de taşınmalı. `remove`'da kaldırılan alanı anlatmanın tek yolu
eski kaydı geçersiz kılmaktır. ADD-only bir katman bu bilgi sınıfını
taşıyamıyor.

Ve tek satırda özeti:
**bilgi %90.9 oranında depoda duruyor, ajan %20'sini kullanabiliyor.**

### Bu, hipotezi hem doğruluyor hem düzeltiyor

Makalede yazmıştım ve ölçmediğimi belirtmiştim:

> "the write path is the problem, and the industry has been optimising
> the read path."

Doğru çıktı — ama sebep yazmanın **kaybetmesi** değil, yazmanın
**karar vermemesi**. Mem0 hiçbir şeyi kaybetmiyor; biriktiriyor ve eskiyi
geçersiz kılmıyor. Ajan çelişkili bağlamda seçim yapmak zorunda kalıyor.

Karşıt kanıt aynı tabloda: "en son şema kazanır" diyen yirmi satırlık
deterministik politika **%100**, tüm geçmişi veren tavan **%95**.
Çelişkisiz az bilgi, çelişkili çok bilgiden iyi.

### Açık kalan ölçüm sorunu

`stored KATI` (13.3%) < `retrieved` (26.7%). Saklanmayan getirilemez, yani
`stored` ölçümü hâlâ eksik: `get_all()` limitsiz çağrılmasına rağmen
`search()` ile aynı kümeyi vermiyor. Bu sütuna dayanan bir iddia
kurulmamalı. Asıl bulgu (`slot kararları`) bu sütundan değil, Mem0'ın
`add()` dönüşünden geliyor — ara bir ölçüte bağlı değil.

## Eski ön bulgu (5 koşu, düzeltilmiş ölçümle)

```
validity=40.0%   stored_kati=0.0%   stored_gevsek=100.0%   retr=20.0%
slot kararlari: {'ADD': 261}
```

**261 yazma kararının hepsi ADD. Sıfır UPDATE, sıfır DELETE.**

Bu, arızanın yerini değiştiriyor:

- Bilgi **kaybolmuyor** — gevşek ölçütte %100, v2 alanları her koşuda depoda.
- Bilgi **getirilemiyor da değil** — katı ölçüt %0 ama validity %40, yani
  ajana ulaşan bir şeyler var.
- Arıza **supersession yokluğu**: şema değiştiğinde eski kayıt güncellenmiyor,
  yanına yenisi ekleniyor. Eski ve yeni yan yana duruyor, hangisinin geçerli
  olduğuna dair işaret yok. Ajan çelişkili bağlamda seçim yapmak zorunda
  kalıyor ve %40'ta kalıyor.

Karşılaştırma: aynı görevde deterministik supersession **%100**.

Bu bulgu makaledeki hipotezi hem doğruluyor hem düzeltiyor.
Yazmıştım: *"the write path is the problem."* Doğru — ama sebep yazmanın
**kaybetmesi** değil, yazmanın **karar vermemesi**.

*(40 koşuluk doğrulama sürüyor.)*

---

## Çoklu model: teşhis modele mi ait, katmana mı?

12 koşu, üç model, hepsi NVIDIA NIM. İki eksen:
**güç** (nemotron 120B → 550B, aynı aile) ve **aile** (nemotron → deepseek).

| kol | nemotron-120B | nemotron-550B | deepseek-v4.1-flash |
|---|---|---|---|
| A no-memory (tavan) | **100%** | **100%** | **100%** |
| C tfidf+recency | **0%** | **0%** | **0%** |
| D supersession | **100%** | **100%** | **100%** |
| E lossy | **0%** | **0%** | **0%** |

**Teşhis tamamen modelden bağımsız.** 4.5 kat güç farkı ve iki ayrı mimari
tabloyu değiştirmiyor. İki okuması var:

- **Güçlü model retrieval'ı kurtarmıyor.** `C` 550B'de de %0 — şema değişikliği
  bağlama hiç ulaşmıyorsa modelin kapasitesi işe yaramıyor.
- **Zayıf model politikayı bozmuyor.** `D` 120B'de de %100 — doğru bağlam
  verildiğinde en küçük model işi yapıyor.

Aradaki 100 puanlık fark modelden değil, katmandan geliyor.

### Bu tabloda düzeltilen bir ölçüm hatası

Ham çıktıda 120B satırı `A 91.7%`, `D 66.7%` görünüyordu. Sebep: servis
hatası (429/503) alan çağrılar "yanlış cevap" gibi paydada bırakılmıştı.
`llm_fail` ile `no_call` ayrımı harness'ta vardı ama bu scriptte
uygulanmamıştı. Servis hatası paydadan çıkarılınca 120B diğerleriyle aynı
hizaya geldi (geçerli koşu: A 11, C 11, D 8, E 10).

Servis dalgalanmasını modelin performansına yazmak, Mem0'da üç kez düşülen
tuzağın aynısı: **altyapının hatası ile ölçülen şeyin hatası karıştırılıyor.**

## Altyapı notu: aynı model, iki farklı uç

Aynı model (`nemotron-3-super-120b`) iki sağlayıcıdan ölçüldü:

| uç | gecikme | limit | gözlem |
|---|---|---|---|
| NVIDIA NIM (doğrudan) | ~2.0s | 40 rpm, günlük sınır yok | sık 429/503 |
| OpenRouter (`:free`) | **1.2s** | 50 istek/gün | tek çağrıda sorunsuz |

OpenRouter'ın ucu daha hızlı ve o anda daha az sıkışıktı, ama günlük 50
istek kotası Mem0'ın tam koşusuna yetmiyor: 15 koşu × (61 olay ÷ 10 batch)
≈ 105 `add`, her biri fact-extraction çağrısı → 120–225 istek.

Denenen diğer `:free` aileler (gemma-4-31b, qwen3.8-27b, glm-5.2) o sırada
"Provider returned error" ile 429 verdi; aile çeşitliliği oradan sağlanamadı.
Çoklu model karşılaştırması bu yüzden NVIDIA içinde yapıldı.

**Ölçümlerin sağlayıcıya duyarlılığı:** çoklu model tablosu her üç modelde de
aynı teşhisi verdi, dolayısıyla bulgunun sağlayıcı kaynaklı olma ihtimali
düşük. Kesilen koşuların sebebi de sağlayıcı değildi — kesilen süreçlerin
çıktısı 10 bayttı, yani uca tek çağrı gitmeden sistem belleği sonlandırmıştı.

## Sınırlar

- **Tek model.** Bulgunun modele mi yoksa katmana mı ait olduğu henüz
  ayrılmadı. Çoklu model karşılaştırması planlandı: güç ekseni
  (`nemotron-120b` → `nemotron-ultra-550b`, aynı aile) ve aile ekseni
  (`deepseek-v4.1-flash`, ayrıca OpenRouter'da `gemma`/`qwen`).
- **Batch=10.** Mem0'a olaylar tek tek değil onarlı gruplar halinde veriliyor.
  40 rpm limitinde 2400 tekil `add` saatler sürüyor. Bu, ürünün inkremental
  konsolidasyon davranışını zayıflatıyor — ve test edilmek istenen kısmen o.
- **Korpus sentetik.** FactConsolidation'a yönelttiğim eleştiri buna da
  geçerli. Farkı: ground truth şema doğrulamasıyla belirleniyor, yorum payı yok.
- **40 koşu** istatistiksel olarak küçük. Uçlarda (%0 vs %100) yeterli,
  ara değerlerde güven aralığı geniş.
- **retry_delta ölçülmedi.** @sukin_s'in önerdiği eksen (başarısızlık sonrası
  memory on/off farkı) henüz kurulmadı.
