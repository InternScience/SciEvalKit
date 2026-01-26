"""
Dataset module with lazy loading for optional dependencies.

This module supports numerous dataset implementations, some with specialized
dependencies. Datasets are only imported when actually used, preventing import
errors for unavailable dependencies.
"""

import warnings
import importlib
import os.path as osp
import numpy as np
import copy as cp
import pandas as pd
from typing import Optional

# Always available - base classes and utilities
from .image_base import img_root_map, ImageBaseDataset
from .utils import *
from ..smp import *

# Lazy load supported_video_datasets when needed
_supported_video_datasets = None


def _get_supported_video_datasets():
    """Lazy load supported_video_datasets from video_dataset_config."""
    global _supported_video_datasets
    if _supported_video_datasets is None:
        from .video_dataset_config import supported_video_datasets
        _supported_video_datasets = supported_video_datasets
    return _supported_video_datasets

# Dataset class registry mapping class names to their module paths
# Organized by module file for easier maintenance
_DATASET_CLASS_REGISTRY = {
    
    'ImageCaptionDataset': '.image_caption',
    
    'ImageYORNDataset': '.image_yorn',
    
    'ImageMCQDataset': '.image_mcq',
    'MMMUDataset': '.image_mcq',
    'CustomMCQDataset': '.image_mcq',
    'MUIRDataset': '.image_mcq',
    'GMAIMMBenchDataset': '.image_mcq',
    'MMERealWorld': '.image_mcq',
    'HRBenchDataset': '.image_mcq',
    'NaturalBenchDataset': '.image_mcq',
    'WeMath': '.image_mcq',
    'MMMUProDataset': '.image_mcq',
    'VMCBenchDataset': '.image_mcq',
    'MedXpertQA_MM_test': '.image_mcq',
    'LEGO': '.image_mcq',
    'VisuLogic': '.image_mcq',
    'CVBench': '.image_mcq',
    'TDBench': '.image_mcq',
    'MicroBench': '.image_mcq',
    'OmniMedVQA': '.image_mcq',
    'MSEarthMCQ': '.image_mcq',
    'VLMBlind': '.image_mcq',
    'SCAM': '.image_mcq',
    '_3DSRBench': '.image_mcq',
    'AffordanceDataset': '.image_mcq',
    'OmniEarthMCQBench': '.image_mcq',
    'XLRSBench': '.image_mcq',
    'TreeBench': '.image_mcq',
    'CVQA': '.image_mcq',
    'TopViewRS': '.image_mcq',

    'MMDUDataset': '.image_mt',

    'ImageVQADataset': '.image_vqa',
    'MathVision': '.image_vqa',
    'OCRBench': '.image_vqa',
    'MathVista': '.image_vqa',
    'LLaVABench': '.image_vqa',
    'LLaVABench_KO': '.image_vqa',
    'VGRPBench': '.image_vqa',
    'MMVet': '.image_vqa',
    'MTVQADataset': '.image_vqa',
    'TableVQABench': '.image_vqa',
    'CustomVQADataset': '.image_vqa',
    'CRPE': '.image_vqa',
    'MathVerse': '.image_vqa',
    'OlympiadBench': '.image_vqa',
    'SeePhys': '.image_vqa',
    'QSpatial': '.image_vqa',
    'VizWiz': '.image_vqa',
    'MMNIAH': '.image_vqa',
    'LogicVista': '.image_vqa',
    'MME_CoT': '.image_vqa',
    'MMSci_Captioning': '.image_vqa',
    'Physics_yale': '.image_vqa',
    'TDBenchGrounding': '.image_vqa',
    'WildDocBenchmark': '.image_vqa',
    'OCR_Reasoning': '.image_vqa',
    'PhyX': '.image_vqa',
    'CountBenchQA': '.image_vqa',
    'ZEROBench': '.image_vqa',
    'Omni3DBench': '.image_vqa',
    'TallyQA': '.image_vqa',
    'MMEReasoning': '.image_vqa',
    'MMVMBench': '.image_vqa',
    'BMMR': '.image_vqa',
    'OCRBench_v2': '.image_vqa',
    'AyaVisionBench': '.image_vqa',
    'SLAKE_EN_TEST': '.image_vqa',

    'CCOCRDataset': '.image_ccocr',
    
    'ImageShortQADataset': '.image_shortqa',
    'PathVQA_VAL': '.image_shortqa',
    'PathVQA_TEST': '.image_shortqa',
        
    'CustomTextMCQDataset': '.text_mcq',
    'TextMCQDataset': '.text_mcq',
    'ProteinLMBench': '.text_mcq',
    
    'BrowseCompZH': '.browsecomp_zh',
    
    'VCRDataset': '.vcr',
    'MMLongBench': '.mmlongbench',
    'DUDE': '.dude',
    'SlideVQA': '.slidevqa',
    'VLRewardBench': '.vl_rewardbench',
    'VLM2Bench': '.vlm2bench',
    'VLMBias': '.vlmbias',
    'Spatial457': '.spatial457',
    'CharXiv': '.charxiv',

    'MMBenchVideo': '.mmbench_video',
    'VideoMME': '.videomme',
    'Video_Holmes': '.video_holmes',

    'MVBench': '.mvbench',
    'MVBench_MP4': '.mvbench',

    'MVTamperBench': '.tamperbench',
    'MIABench': '.miabench',

    'MLVU': '.mlvu',
    'MLVU_MCQ': '.mlvu',
    'MLVU_OpenEnded': '.mlvu',

    'TempCompass': '.tempcompass',
    'TempCompass_Captioning': '.tempcompass',
    'TempCompass_MCQ': '.tempcompass',
    'TempCompass_YorN': '.tempcompass',

    'LongVideoBench': '.longvideobench',
    'MMGenBench': '.mmgenbench',

    'CGBench_MCQ_Grounding_Mini': '.cgbench',
    'CGBench_OpenEnded_Mini': '.cgbench',
    'CGBench_MCQ_Grounding': '.cgbench',
    'CGBench_OpenEnded': '.cgbench',

    'CGAVCounting': '.CGAVCounting.cg_av_counting',

    'MEGABench': '.megabench',
    'MovieChat1k': '.moviechat1k',

    'Video_MMLU_CAP': '.video_mmlu',
    'Video_MMLU_QA': '.video_mmlu',

    'VDC': '.vdc',
    'VCRBench': '.vcrbench',
    'GOBenchDataset': '.gobench',

    'SFE': '.sfebench',
    'EarthSE': '.earthsebench',
    'VisFactor': '.visfactor',
    'OSTDataset': '.ost_bench',

    'EgoExoBench_MCQ': '.EgoExoBench.egoexobench',

    'WorldSense': '.worldsense',
    
    'QBench_Video': '.qbench_video',
    'QBench_Video_MCQ': '.qbench_video',
    'QBench_Video_VQA': '.qbench_video',

    'CMMMU': '.cmmmu',
    'EMMADataset': '.emma',
    'WildVision': '.wildvision',
    'MMMath': '.mmmath',
    'Dynamath': '.dynamath',
    'CreationMMBenchDataset': '.creation',
    'MMAlignBench': '.mmalignbench',

    'OmniDocBench': '.OmniDocBench.omnidocbench',
    
    'MOAT': '.moat',

    'ScreenSpot': '.GUI.screenspot',
    'ScreenSpotV2': '.GUI.screenspot_v2',
    'ScreenSpot_Pro': '.GUI.screenspot_pro',

    'MMIFEval': '.mmifeval',
    'ChartMimic': '.chartmimic',
    'M4Bench': '.m4bench',
    'MMHELIX': '.mmhelix',

    'MedqbenchMCQDataset': '.medqbench_mcq',
    'MedqbenchCaptionDataset': '.medqbench_caption',
    'MedqbenchPairedDescriptionDataset': '.medqbench_paired_description',
    'MaScQA': '.mascqa',

    'SciCode': '.SciCode.scicode',

    'ResearchbenchGenerate': '.Researchbench.generate',
    'ResearchbenchRank': '.Researchbench.rank',
    'ResearchbenchRetrieve': '.Researchbench.retrieve',

    'TRQA': '.trqa',
    'ChemBench': '.ChemBench.chembench',
    'Clima_QA': '.climaqa',
    'PHYSICS': '.PHYSICS.PHYSICS',
    'CMPhysBench': '.CMPhysBench.cmphysbench',

    'SGI_Bench_Experimental_Reasoning': '.SGI_Bench_1_0.experimental_reasoning',
    'SGI_Bench_Deep_Research': '.SGI_Bench_1_0.deep_research',
    'SGI_Bench_Dry_Experiment': '.SGI_Bench_1_0.dry_experiment',
    'SGI_Bench_Wet_Experiment': '.SGI_Bench_1_0.wet_experiment',
    'SGI_Bench_Idea_Generation': '.SGI_Bench_1_0.idea_generation',

    'AstroVisBench': '.AstroVisBench.AstroVisBench',

    'EarthLinkTest': '.earthlink_test',
}


def __getattr__(name: str):
    """
    Lazy loading for dataset classes.
    
    This allows importing dataset classes without requiring their dependencies
    to be installed until the dataset is actually used.
    """
    if name in _DATASET_CLASS_REGISTRY:
        module_path = _DATASET_CLASS_REGISTRY[name]
        try:
            module = importlib.import_module(module_path, package=__name__)
            dataset_class = getattr(module, name)
            # Cache the imported class in the module's namespace
            globals()[name] = dataset_class
            return dataset_class
        except ImportError as e:
            warnings.warn(
                f"Failed to import {name}. "
                f"This dataset may require additional dependencies. "
                f"Error: {e}"
            )
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Show available attributes including lazily-loaded datasets."""
    return list(globals().keys()) + list(_DATASET_CLASS_REGISTRY.keys())


class ConcatDataset(ImageBaseDataset):
    """
    This dataset takes multiple dataset names as input and aggregates them into a single dataset.
    Each single dataset should not have a field named `SUB_DATASET`
    """

    DATASET_SETS = {
        'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
        'MTL_MMBench_DEV': [
            'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
            'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
        ],
        'ScreenSpot_Pro': [
            'ScreenSpot_Pro_Development', 'ScreenSpot_Pro_Creative', 'ScreenSpot_Pro_CAD',
            'ScreenSpot_Pro_Scientific', 'ScreenSpot_Pro_Office', 'ScreenSpot_Pro_OS'
        ],
        'ScreenSpot': ['ScreenSpot_Mobile', 'ScreenSpot_Desktop', 'ScreenSpot_Web'],
        'ScreenSpot_v2': ['ScreenSpot_v2_Mobile', 'ScreenSpot_v2_Desktop', 'ScreenSpot_v2_Web'],
        'M4Bench': ['State_Invariance', 'State_Comparison', 'Spatial_Perception', 
                    'Instance_Comparison', 'Detailed_Difference'],
    }

    def __init__(self, dataset):
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        # The name of the compliation
        self.dataset_name = dataset
        self.datasets = datasets
        for dname in datasets:
            dataset = build_dataset(dname)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        assert np.all([x == TYPES[0] for x in TYPES]), (datasets, TYPES)
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES[0]
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            if 'image' in data:
                data_new = localize_df(data, dname, nproc=16)
                data_all.append(data_new)
            else:
                data_all.append(data)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    def evaluate(self, eval_file, **judge_kwargs):
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        df_all = []
        dict_all = {}
        # One of the vars will be used to aggregate results
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            if isinstance(res, pd.DataFrame):
                res['DATASET'] = [dname] * len(res)
                df_all.append(res)
            elif isinstance(res, dict):
                res = {f'{dname}:{k}': v for k, v in res.items()}
                dict_all.update(res)
            else:
                raise NotImplementedError(f'Unknown result type {type(res)}')

        if len(df_all):
            result = pd.concat(df_all)
            score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
            dump(result, score_file)
            return result
        else:
            score_file = get_intermediate_file_path(eval_file, '_score', 'json')
            dump(dict_all, score_file)
            return dict_all


# ConcatVideoDataset needs to be imported normally as it's used
from .video_concat_dataset import ConcatVideoDataset  # noqa: E402


# Build supported datasets list dynamically
def _get_supported_datasets():
    """Get all supported dataset names from all dataset classes."""
    supported = []
    
    # Add from concat datasets
    supported.extend(ConcatDataset.supported_datasets())
    supported.extend(ConcatVideoDataset.supported_datasets())
    
    # Dynamically check each registered class for supported_datasets
    for class_name in _DATASET_CLASS_REGISTRY:
        try:
            cls = __getattr__(class_name)
            if hasattr(cls, 'supported_datasets'):
                supported.extend(cls.supported_datasets())
        except Exception:
            # If class can't be loaded, skip it
            pass
    
    return supported


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    """Get the type of a dataset (MCQ, VQA, etc.)."""
    # Check ConcatDataset first
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        TYPES = [DATASET_TYPE(dname) for dname in dataset_list]
        assert np.all([x == TYPES[0] for x in TYPES]), (dataset_list, TYPES)
        return TYPES[0]
    
    # Check registered classes
    for class_name in _DATASET_CLASS_REGISTRY:
        try:
            cls = __getattr__(class_name)
            if hasattr(cls, 'supported_datasets') and dataset in cls.supported_datasets():
                return getattr(cls, 'TYPE', default)
        except Exception:
            continue
    
    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')  # noqa: E501
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    """Get the modality of a dataset (IMAGE, VIDEO, TEXT)."""
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    
    # Check ConcatDataset first
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        MODALITIES = [DATASET_MODALITY(dname) for dname in dataset_list]
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (dataset_list, MODALITIES)
        return MODALITIES[0]
    
    # Check registered classes
    for class_name in _DATASET_CLASS_REGISTRY:
        try:
            cls = __getattr__(class_name)
            if hasattr(cls, 'supported_datasets') and dataset in cls.supported_datasets():
                return getattr(cls, 'MODALITY', default)
        except Exception:
            continue
    
    if 'VIDEO' in dataset.upper():
        return 'VIDEO'
    elif 'IMAGE' in dataset.upper():
        return 'IMAGE'
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, **kwargs):
    """
    Build a dataset instance by name.
    
    Args:
        dataset_name: Name of the dataset to build
        **kwargs: Additional arguments to pass to dataset constructor
    
    Returns:
        Dataset instance or None if dataset cannot be built
    """

    # Check registered classes
    for class_name in _DATASET_CLASS_REGISTRY:
        try:
            cls = __getattr__(class_name)
            if hasattr(cls, 'supported_datasets') and dataset_name in cls.supported_datasets():
                return cls(dataset=dataset_name, **kwargs)
        except ImportError as e:
            warnings.warn(
                f"Failed to load dataset class {class_name} for {dataset_name}. "
                f"Dependencies may be missing: {e}"
            )
            continue
        except Exception as e:
            warnings.warn(f"Error loading {class_name}: {e}")
            continue
    
    # Check ConcatDataset
    if dataset_name in ConcatDataset.supported_datasets():
        return ConcatDataset(dataset=dataset_name, **kwargs)
    
    # Check ConcatVideoDataset
    if hasattr(ConcatVideoDataset, 'supported_datasets'):
        if dataset_name in ConcatVideoDataset.supported_datasets():
            return ConcatVideoDataset(dataset=dataset_name, **kwargs)
    
    # Check supported_video_datasets from video_dataset_config (lazy loaded)
    supported_video_datasets = _get_supported_video_datasets()
    if dataset_name in supported_video_datasets:
        return supported_video_datasets[dataset_name](**kwargs)

    # Try custom dataset fallback
    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')
    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    # Lazy load custom dataset classes
    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            CustomMCQDataset = __getattr__('CustomMCQDataset')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        else:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom Text MCQ dataset. ')
            CustomTextMCQDataset = __getattr__('CustomTextMCQDataset')
            return CustomTextMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        CustomVQADataset = __getattr__('CustomVQADataset')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


def infer_dataset_basename(dataset_name):
    """Infer the base name of a dataset from its full name."""
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


# Cache for supported datasets (computed on demand)
_SUPPORTED_DATASETS_CACHE: Optional[list] = None


def get_supported_datasets():
    """Get list of all supported datasets (cached)."""
    global _SUPPORTED_DATASETS_CACHE
    if _SUPPORTED_DATASETS_CACHE is None:
        _SUPPORTED_DATASETS_CACHE = _get_supported_datasets()
    return _SUPPORTED_DATASETS_CACHE


# For backwards compatibility, provide SUPPORTED_DATASETS as a property
class _SupportedDatasets:
    def __iter__(self):
        return iter(get_supported_datasets())
    
    def __contains__(self, item):
        return item in get_supported_datasets()
    
    def __len__(self):
        return len(get_supported_datasets())


SUPPORTED_DATASETS = _SupportedDatasets()


__all__ = [
    'build_dataset',
    'img_root_map',
    'build_judge',
    'extract_answer_from_item',
    'prefetch_answer',
    'DEBUG_MESSAGE',
    'DATASET_TYPE',
    'DATASET_MODALITY',
    'infer_dataset_basename',
    'get_supported_datasets',
    'SUPPORTED_DATASETS',
    'ConcatDataset',
    'ConcatVideoDataset',
    'ImageBaseDataset',
]
