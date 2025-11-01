import { ChatOption } from '../../lib/types';

interface OptionButtonsProps {
  options: ChatOption[];
  onSelect: (option: ChatOption) => void;
  disabled?: boolean;
}

export default function OptionButtons({ options, onSelect, disabled = false }: OptionButtonsProps) {
  return (
    <div className="flex flex-col space-y-2 mt-3">
      {options.map((option, index) => (
        <button
          key={index}
          onClick={() => onSelect(option)}
          disabled={disabled}
          className="text-left p-3 bg-white border border-gray-200 rounded-lg hover:bg-green-50 hover:border-green-300 transition-all duration-200 text-sm font-medium text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}