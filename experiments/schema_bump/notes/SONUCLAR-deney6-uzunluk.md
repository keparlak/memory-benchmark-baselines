# Deney 6 — Depo büyüdükçe retrieval boğuluyor mu?

Tarih: 30 Eylül 2026
Kod: `deney6_uzunluk.py`, `deney6_analiz.py`, veri: `deney6_sonuclar/` (24 JSON, ham metinler dahil)
Ürün: **mem0ai 2.1.0 açık kaynak** · Model: `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM, reasoning kapalı)

## Soru

@kartikb753, 23 Eylül:

> "An append-only memory can never say this is gone, only this is newer. The
> question is whether anything downstream ever garbage collects, or if the store
> just grows until retrieval drowns."

## Neden adil bir soru — Mem0'ın kendi iddiası

mem0 v2.0.0 (2026-04-14) tasarım gereği ADD-only:

> "Single-Pass Extraction: Replaced 2-LLM-call pipeline with additive extraction
> … Memories accumulate via linked_memory_ids: no more UPDATE/DELETE events."
> — [docs.mem0.ai/changelog/sdk](https://docs.mem0.ai/changelog/sdk)

> "When information changes, the new fact is stored alongside the old one.
> Retrieval handles ranking: the most relevant, current information surfaces first."
> — [docs.mem0.ai/migration/oss-v2-to-v3](https://docs.mem0.ai/migration/oss-v2-to-v3)

Kaynak kodda (`mem0/utils/scoring.py`, `score_and_rank`) sıralamanın girdileri:
anlamsal benzerlik, BM25, varlık bonusu. **Zaman terimi yok.** Varlık bonusu
aynı varlığa bağlı tüm kayıtlara eşit uygulanıyor ve kayıt sayısıyla küçülüyor
(`1/(1+0.001·(n−1)²)`). Zamansal arama (`reference_date`) ve sönümleme
(`decay`) yalnızca ücretli Platform'da — **Platform test edilmedi.**

Bu deney belgedeki iddiayı ölçüyor: depo büyüdükçe güncel bilgi öne çıkmaya
devam ediyor mu?

## Tasarım

- **Tek değişken uzunluk:** 20 / 60 / 120 / 200 adım. Mutasyon tipi sabit
  (her uzunlukta aynı 6: rename ×2, add_required ×2, type_change ×2).
- **İki arama sorgusu:**
  - *schema:* `"{tool} current schema arguments"` — harness'in Deney 5'teki sorgusu
  - *görev:* `"call {tool} to {task}"` — ajanın gerçekte isteyeceği şey
- **Ajana giden bağlam:** ilk 20 kayıt (`top_k=20`).
- **Kontroller her uzunlukta, aynı ajanla:** tüm geçmiş bağlamda; deterministik supersession.
- Servis hatası paydaya girmez. Her koşu ham depo ve sıralama metinleriyle kaydedilir.

## Sonuç

| olay | depo | değişiklik kaydı var | **Mem0 görev sorgusu** | Mem0 schema sorgusu | tüm geçmiş | supersession |
|---|---|---|---|---|---|---|
| 20 | 16 | 5/6 | **83%** | 83% | 100% | 100% |
| 60 | 47 | 3/6 | **67%** | 67% | 100% | 83% |
| 120 | 103 | 3/6 | **17%** | 50% | 100% | 100% |
| 200 | 182 | 3/6 | **33%** | 67% | 100% | 100% |

Birleştirilmiş (n=12 / grup, Fisher exact, iki yönlü):

| | ≤60 olay | ≥120 olay | p |
|---|---|---|---|
| **Mem0, görev sorgusu** | 9/12 (75%) | 3/12 (25%) | **0.039** |
| Mem0, schema sorgusu | 9/12 (75%) | 7/12 (58%) | 0.667 |
| Tüm geçmiş bağlamda | 12/12 | 12/12 | 1.000 |

## Kartik'in sorusuna cevap

**GC yok.** 24 koşuda 2.086 yazma kararı, hepsi ADD. Depo geçmişi neredeyse
birebir izliyor: olay başına 0.78 → 0.79 → 0.86 → 0.91 kayıt.

**Retrieval boğuluyor — ama sadece gerçekçi sorguyla görünüyor.**
Ajanın kendi görevi sorgu olduğunda:

| olay | değişiklik kaydının medyan sırası | ilk 20'de eski kanıt | ilk 20'de yeni kanıt |
|---|---|---|---|
| 20 | 5 | 5.3 | 6.3 |
| 60 | 28 | 8.7 | 10.3 |
| 120 | 47 | 10.3 | 9.0 |
| 200 | 0* | 13.0 | 6.2 |

\* 200'de kayıt 3 koşuda var: ikisinde 1. sırada, birinde ilk 100'de bile yok.

≥60 olayda değişiklik kaydının saklandığı 9 koşunun **5'inde** kayıt görev
sorgusunda ilk 20'nin dışına düştü. 200 olayda ajana verilen 20 kaydın ortalama
13'ü eski şemayı gösteriyor.

## İki ayrı mekanizma

**1. Yazma:** Mem0'ın tek geçişli çıkarımı, şema değişikliği bildirimini her
zaman ayrı bir bilgi olarak saklamıyor; bazen yalnızca değişiklik sonrası
çağrıları kaydediyor. Değişiklik kaydı varsa (schema sorgusuyla) **14/14**
başarılı; yoksa **2/10**. Saklanma oranı: 5/6 (20 olay) → 3/6 (60, 120, 200).

Ham metinden örnek — kayıt var:
> "The create_ticket function signature was updated: 'priority' parameter
> removed and replaced with 'severity' (int), new signature is …"

Kayıt yok (add_required, 120 olay): depoda "team" alanını içeren 36 kayıt var,
hepsi çağrı ("User created ticket … and team refund-280"); "team artık zorunlu"
diyen kayıt sıfır.

**2. Okuma (Kartik'in sorusu):** Kayıt saklansa bile, zamana kör sıralamada
eski çağrı kayıtlarıyla yalnızca metin benzerliğiyle yarışıyor. Depo büyüdükçe
gerçekçi sorguda aşağı düşüyor.

## Harness'e karşı bir düzeltme

Deney 5'in "schema" sorgusu **Mem0'ın lehineydi**: içinde "schema" geçtiği için
değişikliği anlatan kayda doğal olarak yakın. Bu sorguda kayıt her uzunlukta
1. sırada çıktı ve boğulma görünmedi (p=0.667). Gerçekçi görev sorgusu farkı
ortaya çıkardı. Deney 5'teki Mem0 sayıları bu yüzden Mem0'a karşı cömert
olabilir.

## Kontrollerin söylediği

Aynı ajan, aynı olaylar, **200 olayın tamamı bağlamda: 24/24.**
Deterministik supersession: 23/24. Yani 200 olayda ajanın kendisi sorun
yaşamıyor; kayıp hafıza katmanında.

## Sınırlar

- **n küçük:** uzunluk başına 6 koşu. Birleştirilmiş fark p=0.039 — anlamlı ama
  sınırda. 120 → 200 arasındaki hafif toparlanma (17% → 33%) gürültü içinde.
- **Tek ürün sürümü:** mem0 OSS 2.1.0. Ücretli Platform'daki zamansal arama ve
  sönümleme test edilmedi; orada sonuç farklı olabilir.
- **Tek model**, sentetik korpus, `batch=10` (olaylar tek tek değil onarlı verildi).
- **Yazma hatası:** bir koşuda (200 olay, add_required m3) Mem0 modelin
  çıkarım çıktısını ayrıştıramadı (`'int' object has no attribute 'get'`),
  10 olay depoya girmedi. Koşu tabloda duruyor, işaretli.
- **Sınıflandırıcı:** değişiklik kaydı tespiti sentetik korpusun bir özelliğine
  dayanıyor (çağrı kayıtlarında hep `kelime-ddd` değeri var). `due`/`due_ts`
  mutasyonunda (m6) "eski kanıt" sayılmadı: `due` İngilizce kelime de ("due
  date"), kelime sınırlı eşleşme ayırt edemiyor.
- **İlk koşu turu atıldı:** ilk sınıflandırıcı alt-dize eşleşmesi kullanıyordu
  ve `due`'yu `due_ts` içinde buluyordu (imkânsız `15/20` sonuçlar). Düzeltilip
  24 koşunun hepsi aynı kodla baştan koşuldu; eski sonuçlar
  `deney6_sonuclar_v0_eski_siniflandirici/` altında duruyor.
